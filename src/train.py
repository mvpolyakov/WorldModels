import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sim.build_world import load_world_builder
from training.trajectories import (
    TrajectorySliceDataset,
    create_trajectory_sequence_dataset,
)
from world_models import LatentStateModel

WARMUP_LENGTH = 3  # Min 2 frames for change detection since includes the frame from which we predict; three for acceleration.
PREDICTION_HORIZON = 3


def _compute_observation_states(
    model: LatentStateModel,
    observed_positions: torch.Tensor,
    scene_features: torch.Tensor,
    padding_frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Traverse observations chronologically, retaining state across padded frames.

    B is trajectory count, T is padded frame count, L is latent width, and F is
    scene-feature width. Gradients remain connected through the entire sequence.

    @param observed_positions: XYZ observations (B, T, 3), with finite padding.
    @param scene_features: Static scene features (B, F).
    @param padding_frame_mask: Boolean tensor (B, T); True marks trailing padding.

    @return: Observation-updated states (B, T, L). Padded entries repeat the
        trajectory's last observed state and are excluded from losses.
    """
    current_state = model.initialize_state(
        observed_positions.shape[0],
        device=observed_positions.device,
        dtype=observed_positions.dtype,
    )
    observation_states = []
    # step through each frame of the trajectories - whole batch at once - 
    for frame_index in range(observed_positions.shape[1]):
        # The first observation initializes state at its own time; later frames
        # require one transition before incorporating the new observation.
        prior_state = current_state if frame_index == 0 else model.predict_state(current_state)
        
        observation_embedding = model.encode_observation(
            observed_positions[:, frame_index], scene_features
        )
        updated_state = model.update_state(prior_state, observation_embedding)
        current_state = torch.where(
            padding_frame_mask[:, frame_index].unsqueeze(-1),
            current_state,
            updated_state,
        )
        observation_states.append(current_state)

    return torch.stack(observation_states, dim=1)

def _compute_consistency_error_sum(
    predicted_states: torch.Tensor,
    observation_states: torch.Tensor,
    padding_frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Score predicted latents against detached observation-informed targets.

    B is trajectory count and L is latent width. Only this use of the target is
    detached; reconstruction and future forecasts can still train its encoder.

    @param predicted_states: Autonomous predictions (B, L).
    @param observation_states: Observation-updated targets at the same time (B, L).
    @param padding_frame_mask: Boolean tensor (B,); True excludes a trajectory.

    @return: Scalar sum of squared latent errors over non-padding entries.
    """
    return torch.sum(
        (predicted_states - observation_states.detach()) ** 2
        * (~padding_frame_mask).unsqueeze(-1)
    )

def _compute_batch_losses(
    model: LatentStateModel,
    observed_positions: torch.Tensor,
    scene_features: torch.Tensor,
    padding_frame_mask: torch.Tensor,
):
    """Compute reconstruction and autonomous forecast losses for a padded batch.

    B is trajectory count, T is padded frame count, L is latent width, and F is
    scene-feature width. Each forecast starts after WARMUP_LENGTH observations
    and advances up to PREDICTION_HORIZON frames without observation updates.

    @param observed_positions: XYZ observations (B, T, 3), with finite padding.
    @param scene_features: Static scene features (B, F).
    @param padding_frame_mask: Boolean tensor (B, T); True marks trailing padding.

    @return: Scalar tensors position_loss, reconstruction_loss, consistency_loss,
        and latent_std; integer position_element_count, reconstruction_element_count,
        consistency_element_count, and observed_frame_count for epoch aggregation.
        Forecast losses are zero when no frames remain after warm-up.
    """
    batch_size, padded_frame_count, position_size = observed_positions.shape

    # Observation branch first since the autonomous rollouts depend on it
    # prediction_horizon doesn't play a role here - observation updates computed once per frame

    observation_states = _compute_observation_states(
        model, observed_positions, scene_features, padding_frame_mask
    )
    observed_frame_count = int((~padding_frame_mask).sum().item())

    # Decode each observation-updated state to train the encoder and updater.
    # Flatten trajectory/time axes for the decoder, then restore the frame layout.
    reconstructed_positions = model.decode_state(
        observation_states.reshape(batch_size * padded_frame_count, model.latent_size)
    ).reshape(batch_size, padded_frame_count, position_size)

    reconstruction_error_sum = torch.sum(
        (reconstructed_positions - observed_positions) ** 2
        * (~padding_frame_mask).unsqueeze(-1)
    )
    reconstruction_element_count = observed_frame_count * position_size


    # Autonomous rollout branch

    # init to a scalar tensor that takes a gradient (useful if n=0)
    position_error_sum = observed_positions.new_zeros(())
    consistency_error_sum = observed_positions.new_zeros(())

    forecast_frame_count = 0

    # A forecast starts at the last observed frame of its warm-up. Its state
    # remains connected to that history, but receives no subsequent observations.
    for forecast_start_frame in range(WARMUP_LENGTH - 1, padded_frame_count - 1):
        predicted_state = observation_states[:, forecast_start_frame]
        forecast_stop_frame = min(
            padded_frame_count,
            forecast_start_frame + PREDICTION_HORIZON + 1,
        )
        for target_frame in range(forecast_start_frame + 1, forecast_stop_frame):
            predicted_state = model.predict_state(predicted_state)
            predicted_position = model.decode_state(predicted_state)
            forecast_padding_mask = (
                padding_frame_mask[:, forecast_start_frame]
                | padding_frame_mask[:, target_frame]
            )
            position_error_sum = position_error_sum + torch.sum(
                (predicted_position - observed_positions[:, target_frame]) ** 2
                * (~forecast_padding_mask).unsqueeze(-1)
            )
            # avoid += because that would modify value in place -- for autograd that
            # can be an issue if the earlier value is needed later (not here)
            consistency_error_sum = (
                consistency_error_sum
                + _compute_consistency_error_sum(
                    predicted_state,
                    observation_states[:, target_frame],
                    forecast_padding_mask,
                )
            )
            forecast_frame_count += int((~forecast_padding_mask).sum().item())

    # Average over scored scalar values, not padded frames or trajectory count.
    # A frame can be scored by several forecasts starting at different times.
    position_element_count = forecast_frame_count * position_size
    consistency_element_count = forecast_frame_count * model.latent_size
    observed_states = observation_states[~padding_frame_mask]
    return {
        "position_loss": position_error_sum / max(position_element_count, 1),
        "reconstruction_loss": reconstruction_error_sum / max(reconstruction_element_count, 1),
        "consistency_loss": consistency_error_sum / max(consistency_element_count, 1),
        "latent_std": observed_states.std(dim=0, unbiased=False).mean(),
        "position_element_count": position_element_count,
        "reconstruction_element_count": reconstruction_element_count,
        "consistency_element_count": consistency_element_count,
        "observed_frame_count": observed_frame_count,
    }


def _process_epoch_pass(
    model: LatentStateModel,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    position_weight: float,
    reconstruction_weight: float,
    consistency_weight: float,
) -> dict[str, float]:
    """Run one training or validation pass over trajectory batches.

    Supplying an optimizer enables gradient updates; None computes validation
    metrics only. Losses are weighted by their scored scalar counts; latent_std
    is a frame-weighted mean of batch statistics, not a global standard deviation.

    @param model: LatentStateModel being trained or evaluated.
    @param data_loader: Loader yielding padded trajectory batches and masks.
    @param device: Torch device receiving each batch.
    @param optimizer: Optimizer for training, or None for validation.
    @param position_weight: Multiplier for autonomous position-prediction MSE.
    @param reconstruction_weight: Multiplier for observation reconstruction MSE.
    @param consistency_weight: Multiplier for detached-target latent MSE.

    @return: Dictionary of mean losses, latent standard deviation, and total loss.
    """
    metric_sums = {
        "position_loss": 0.0,
        "reconstruction_loss": 0.0,
        "consistency_loss": 0.0,
        "latent_std": 0.0,
    }
    metric_counts = {
        "position_loss": 0,
        "reconstruction_loss": 0,
        "consistency_loss": 0,
        "latent_std": 0,
    }
    print('process_epoch_pass')
    for observed_positions, scene_features, _, padding_frame_mask in data_loader:
        observed_positions = observed_positions.to(device)
        scene_features = scene_features.to(device)
        padding_frame_mask = padding_frame_mask.to(device)
        if optimizer is not None:
            optimizer.zero_grad()

        losses = _compute_batch_losses(
            model,
            observed_positions,
            scene_features,
            padding_frame_mask, # which frame for each trajectory are real and which are just padding at end
        )
        total_loss = (
            position_weight * losses["position_loss"]
            + reconstruction_weight * losses["reconstruction_loss"]
            + consistency_weight * losses["consistency_loss"]
        )
        if not torch.isfinite(total_loss):
            pass_name = "Validation" if optimizer is None else "Training"
            raise RuntimeError(f"{pass_name} loss became non-finite.")
        if optimizer is not None:
            total_loss.backward()
            optimizer.step()

        counts = {
            "position_loss": losses["position_element_count"],
            "reconstruction_loss": losses["reconstruction_element_count"],
            "consistency_loss": losses["consistency_element_count"],
            "latent_std": losses["observed_frame_count"],
        }
        for name in metric_sums:
            metric_sums[name] += losses[name].item() * counts[name]
            metric_counts[name] += counts[name]

    metrics = {
        name: metric_sums[name] / max(metric_counts[name], 1)
        for name in metric_sums
    }
    metrics["total_loss"] = (
        position_weight * metrics["position_loss"]
        + reconstruction_weight * metrics["reconstruction_loss"]
        + consistency_weight * metrics["consistency_loss"]
    )
    return metrics


def _roll_out_model(
    model,
    observed_positions,
    scene_features,
    prediction_count,
    device,
):
    """Build a latent from observed frames, then predict without observations (autonomous rollout)
    This imitates the eval script can be used for evaluation of long-term rollout accuracy.
    NOT USED IN TRAINING.    

    The first observation updates a zero latent at its current frame. Each later
    warm-up observation first advances the latent once and then updates it. After
    the final observation, the function repeatedly predicts and decodes future
    states; predicted positions are never fed back as observations.

    @param model: LatentStateModel used for observation updates and prediction.
    @param observed_positions: Warm-up positions shaped (B, T, 3) or (T, 3).
    @param scene_features: Static scene features shaped (B, F) or (F,).
    @param prediction_count: Number of future frames to predict autonomously.
    @param device: Torch device used for the rollout.

    @return: Predicted absolute positions with shape (B, prediction_count, 3).
    """

    # Format args and init

    observed_positions = torch.as_tensor(
        observed_positions,
        dtype=torch.float32,
        device=device,
    )
    if observed_positions.ndim == 2:
        observed_positions = observed_positions.unsqueeze(0)

    scene_features = torch.as_tensor(
        scene_features,
        dtype=torch.float32,
        device=device,
    )
    if scene_features.ndim == 1:
        scene_features = scene_features.unsqueeze(0)
    if observed_positions.ndim != 3 or observed_positions.shape[2] != 3:
        raise ValueError("observed_positions must have shape (batch, frames, 3)")
    if observed_positions.shape[1] == 0:
        raise ValueError("At least one warm-up observation is required.")

    state = model.initialize_state(
        observed_positions.shape[0],
        device=device,
        dtype=observed_positions.dtype,
    )

    # process warm up period to build up latent state
    
    for frame_index in range(observed_positions.shape[1]):
        state = model.encode_predict_update_helper(
            state,
            observed_positions[:, frame_index],
            scene_features,
            advance=frame_index > 0,
        )

    # actually do autonomous rollout
    
    predicted_positions = []
    for _ in range(prediction_count):
        predicted_position, state = model(state)
        predicted_positions.append(predicted_position)

    if len(predicted_positions) == 0:
        return observed_positions.new_empty((observed_positions.shape[0], 0, 3))

    return torch.stack(predicted_positions, dim=1)


@torch.inference_mode()
def _compute_full_trajectory_rollouts(model, validation_dataset, device):
    """Predict every validation trajectory after its observation warm-up.

    Run after training finishes, this imitates the eval script and can be
    used for evaluation of long-term rollout accuracy and detail error analysis.
    NOT USED IN TRAINING.

    @param model: LatentStateModel used for autonomous prediction.
    @param validation_dataset: Dataset containing whole trajectories and scene features.
    @param device: Torch device used for prediction and error calculation.

    @return: One dictionary per trajectory containing actual and predicted XYZ
        positions plus position, velocity, and velocity-innovation errors.
    """
    trajectory_rollouts = []

    history_length = model.get_required_history()
    for trajectory_positions, trajectory_features in zip(
        validation_dataset.trajectories,
        validation_dataset.scene_feature_tensor,
    ):
        actual_positions = torch.from_numpy(trajectory_positions).to(device)
        scene_features = trajectory_features.unsqueeze(0).to(device)

        predicted_positions = _roll_out_model(
            model,
            actual_positions[:history_length],
            scene_features,
            len(actual_positions) - history_length,
            device,
        )[0]

        actual_future_positions = actual_positions[history_length:]
        position_errors = predicted_positions - actual_future_positions

        previous_predicted_positions = torch.cat(
            [actual_positions[history_length - 1:history_length], predicted_positions[:-1]]
        )
        previous_actual_positions = actual_positions[history_length - 1:-1]
        velocity_errors = (
            predicted_positions - previous_predicted_positions
            - (actual_future_positions - previous_actual_positions)
        )
        velocity_error_innovations = torch.diff(
            velocity_errors,
            dim=0,
            prepend=torch.zeros_like(velocity_errors[:1]),
        )

        trajectory_rollouts.append(
            {
                "actual_positions": actual_future_positions.cpu().numpy(),
                "predicted_positions": predicted_positions.cpu().numpy(),
                "position_errors": position_errors.cpu().numpy(),
                "velocity_errors": velocity_errors.cpu().numpy(),
                "velocity_error_innovations": velocity_error_innovations.cpu().numpy(),
            }
        )

    return trajectory_rollouts

def train_model(
    model,
    train_data_loader,
    validation_data_loader,
    config,
    device,
):
    """Train the latent model and evaluate full autonomous validation rollouts.

    @param model: LatentStateModel to optimize.
    @param train_data_loader: Loader of padded training trajectory slices.
    @param validation_data_loader: Loader of fixed validation trajectory slices.
    @param config: Learning rate, epoch count, and loss weights.
    @param device: Torch device used for training and validation.

    @return: Per-epoch training metrics, per-epoch validation metrics, and final
        full-trajectory validation rollout dictionaries.
    """
    optimizer = torch.optim.Adam(model.parameters(), config["lr"])
    training_metrics = []
    validation_metrics = []
    validation_trajectory_rollouts = []

    for epoch in range(config["epochs"]):
        train_data_loader.dataset.resample_slices(epoch)
        model.train()
        training_metrics.append(
            _process_epoch_pass(
                model,
                train_data_loader,
                device,
                optimizer,
                config["position_weight"],
                config["reconstruction_weight"],
                config["consistency_weight"],
            )
        )

        model.eval()
        with torch.inference_mode():
            validation_metrics.append(
                _process_epoch_pass(
                    model,
                    validation_data_loader,
                    device,
                    None,
                    config["position_weight"],
                    config["reconstruction_weight"],
                    config["consistency_weight"],
                )
            )
            if epoch == config["epochs"] - 1:
                validation_trajectory_rollouts = _compute_full_trajectory_rollouts(
                    model,
                    validation_data_loader.dataset.source_dataset,
                    device,
                )

        if epoch % 2 == 0:
            print("epoch", epoch)

    return training_metrics, validation_metrics, validation_trajectory_rollouts

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a latent-state world model.")
    parser.add_argument("--world-builder", default="opposing-ramps")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trajectory-batch-size", type=int, default=4)
    parser.add_argument("--slices-per-trajectory", type=int, default=4)
    parser.add_argument("--max-slice-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)  # not dynamics model
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--observation-embedding-size", type=int, default=32)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
   # parser.add_argument("--output-dir", default="artifacts/phase1_0/run_001")
    return parser.parse_args()

def _store_validation_rollouts(validation_trajectory_rollouts, output_dir):
    validation_rollout_lengths = np.asarray(
        [
            len(trajectory_rollout["position_errors"])
            for trajectory_rollout in validation_trajectory_rollouts
        ]
    )
    max_validation_rollout_length = validation_rollout_lengths.max()
    saved_validation_rollouts = {}
    for key in validation_trajectory_rollouts[0]:
        values = np.full(
            (
                len(validation_trajectory_rollouts),
                max_validation_rollout_length,
                3,
            ),
            np.nan,
            dtype=np.float32,
        )
        for trajectory_index, trajectory_rollout in enumerate(
            validation_trajectory_rollouts
        ):
            trajectory_values = trajectory_rollout[key]
            values[trajectory_index, :len(trajectory_values)] = trajectory_values
        saved_validation_rollouts[key] = values
    np.savez(
        output_dir / "validation_rollouts.npz",
        rollout_lengths=validation_rollout_lengths,
        **saved_validation_rollouts,
    )

def main() -> None:
    args = get_args()

    import genesis as gs

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")

    genesis_backend = gs.cpu if args.backend == "cpu" else gs.gpu
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        gs.init(backend=genesis_backend, seed=args.seed)
    else:
        gs.init(backend=genesis_backend)

    random_generator = np.random.default_rng(None if args.seed == 0 else args.seed)
    sampling_seeds = random_generator.integers(
        0,
        np.iinfo(np.int64).max,
        size=2,
        dtype=np.int64,
    )
    training_sampling_seed = int(sampling_seeds[0])
    validation_sampling_seed = int(sampling_seeds[1])

    world_builder = load_world_builder(args.world_builder)

    trajectories, training_sequence_dataset, validation_sequence_dataset = (
        create_trajectory_sequence_dataset(
            world_builder,
            args.seed,
            args.steps,
            args.trajectories,
        )
    )

    # Genesis sets PyTorch's default device to its backend during initialization.
    # Keep DataLoader indices and shuffle permutations on CPU.
    # must be here, after gs.init and trajectory creation
    torch.set_default_device("cpu")
    training_dataset = TrajectorySliceDataset(
        training_sequence_dataset,
        slices_per_trajectory=args.slices_per_trajectory,
        max_slice_length=args.max_slice_length,
        warm_up_length=WARMUP_LENGTH,
        sampling_seed=training_sampling_seed,
    )
    validation_dataset = TrajectorySliceDataset(
        validation_sequence_dataset,
        slices_per_trajectory=args.slices_per_trajectory,
        max_slice_length=args.max_slice_length,
        warm_up_length=WARMUP_LENGTH,
        sampling_seed=validation_sampling_seed,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.trajectory_batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.trajectory_batch_size,
        shuffle=False,
    )
    model = LatentStateModel(
        latent_size=args.latent_size,
        observation_embedding_size=args.observation_embedding_size,
        hidden_size=args.hidden_size,
        scene_feature_size=world_builder.scene_feature_size,
        warm_up_length=WARMUP_LENGTH,
    ).to(device)
    config = {
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "position_weight": args.position_weight,
        "reconstruction_weight": args.reconstruction_weight,
        "consistency_weight": args.consistency_weight,
    }
    training_metrics, validation_metrics, validation_trajectory_rollouts = (
        train_model(
            model,
            training_loader,
            validation_loader,
            config,
            device,
        )
    )
    checkpoint = {
        "model_type": "LatentStateModel",
        "world_builder": args.world_builder,
        "latent_size": model.latent_size,
        "observation_embedding_size": model.observation_embedding_size,
        "hidden_size": model.hidden_size,
        "scene_feature_size": model.scene_feature_size,
        "warm_up_length": model.warm_up_length,
        "prediction_horizon": PREDICTION_HORIZON,
        "state_dict": model.state_dict(),
        "training_metadata": {
            "prediction_horizon": PREDICTION_HORIZON,
            "position_weight": args.position_weight,
            "reconstruction_weight": args.reconstruction_weight,
            "consistency_weight": args.consistency_weight,
            "trajectory_batch_size": args.trajectory_batch_size,
            "slices_per_trajectory": args.slices_per_trajectory,
            "max_slice_length": args.max_slice_length,
            "training_sampling_seed": training_sampling_seed,
            "validation_sampling_seed": validation_sampling_seed,
            "validation_slice_metadata": (
                validation_dataset.get_slice_metadata().tolist()
            ),
        },
    }
    saved_metrics = {
        "training": training_metrics,
        "validation": validation_metrics,
    }

    output_dir = Path('artifacts/runs')
    output_dir.mkdir(parents=True, exist_ok=True)

    _store_validation_rollouts(validation_trajectory_rollouts, output_dir)

    trajectory_lengths = np.asarray([len(trajectory) for trajectory in trajectories])
    saved_positions = np.full(
        (len(trajectories), trajectory_lengths.max(), 3),
        np.nan,
        dtype=np.float32,
    )
    for trajectory_index, trajectory in enumerate(trajectories):
        saved_positions[trajectory_index, :len(trajectory)] = trajectory

    saved_scene_features = torch.cat(
        [
            training_sequence_dataset.scene_feature_tensor,
            validation_sequence_dataset.scene_feature_tensor,
        ]
    ).cpu().numpy()
    np.savez(
        output_dir / "trajectory.npz",
        positions=saved_positions,
        trajectory_lengths=trajectory_lengths,
        scene_features=saved_scene_features,
    )
    torch.save(checkpoint, output_dir / "checkpoint.tar")
    (output_dir / "metrics.json").write_text(
        json.dumps(saved_metrics, indent=2)
    )
    print(f"Saved training artifacts to {output_dir}")


if __name__ == "__main__":
    main()
