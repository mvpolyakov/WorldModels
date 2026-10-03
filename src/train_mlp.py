import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from sim.build_world import load_world_builder
from training import (
    PositionHistoryDataset,
    SimulationTrajectoryCollector,
    create_trajectory_dataset
)
from world_models import PositionHistoryMLP

HISTORY_WINDOW = 5
PREDICTION_HORIZON = 5
velocity_noise_probability = 0.75

def _add_random_walk_velocity_noise(
    histories,
    velocity_innovation_std,
):
    interval_count = histories.shape[1] - 1
    if (
        velocity_innovation_std == 0.0
        or velocity_noise_probability == 0.0
        or interval_count == 0
    ):
        return histories

    velocity_innovations = torch.randn_like(histories[:, 1:])
    velocity_innovations *= velocity_innovation_std
    velocity_noise = torch.cumsum(velocity_innovations, dim=1)
    position_noise = torch.cumsum(velocity_noise, dim=1)
    position_noise = torch.cat(
        [torch.zeros_like(histories[:, :1]), position_noise],
        dim=1,
    )
    if velocity_noise_probability < 1.0:
        noisy_samples = torch.rand(
            (histories.shape[0], 1, 1),
            device=histories.device,
        ) < velocity_noise_probability
        position_noise *= noisy_samples
    return histories + position_noise


def _process_epoch_pass(
    model,
    data_loader,
    device,
    optimizer,
    input_velocity_innovation_std=0.0,
):
    total_loss = 0.0
    sample_count = 0

    for batch in data_loader:
        histories, targets, scene_features = batch
        histories = histories.to(device)
        targets = targets.to(device)
        scene_features = scene_features.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
            histories = _add_random_walk_velocity_noise(
                histories,
                input_velocity_innovation_std,
            )

        # Predict multiple steps into the future progressively adding predictions into
        # history for more informative loss calc
        predictions = []
        rollout_history = histories
        for _ in range(targets.shape[1]):
            predicted_position = model(rollout_history, scene_features)
            predictions.append(predicted_position)
            rollout_history = torch.cat(
                [rollout_history[:, 1:], predicted_position.unsqueeze(1)],
                dim=1,
            )
        predictions = torch.stack(predictions, dim=1)
        loss = torch.mean((predictions - targets) ** 2)
        if not torch.isfinite(loss):
            error = ('Validation' if optimizer is None else 'Training') + "loss became non-finite."
            raise RuntimeError(error)
        
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        count = len(histories)
        total_loss += loss.item() * count
        sample_count += count

    return total_loss / sample_count


@torch.inference_mode()
def _evaluate_full_trajectory_rollouts(model, validation_dataset, device):
    trajectory_rollouts = []

    for trajectory_dataset in validation_dataset.datasets:
        actual_positions = torch.from_numpy(trajectory_dataset.positions).to(device)
        history_length = trajectory_dataset.X.shape[1]
        rollout_history = actual_positions[:history_length].unsqueeze(0)
        scene_features = torch.from_numpy(
            trajectory_dataset.scene_features
        ).unsqueeze(0).to(device)

        predicted_positions = []
        for _ in range(history_length, len(actual_positions)):
            predicted_position = model(rollout_history, scene_features)
            predicted_positions.append(predicted_position[0])
            rollout_history = torch.cat(
                [rollout_history[:, 1:], predicted_position.unsqueeze(1)],
                dim=1,
            )

        predicted_positions = torch.stack(predicted_positions)
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

def train_model(model,
                train_data_loader,
                validation_data_loader,
                config,
                device):

    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    training_errors = []
    validation_errors = []
    validation_trajectory_rollouts = []

    for epoch in range(config['epochs']):
        model.train()

        mean_error =_process_epoch_pass(model, 
                                        train_data_loader, 
                                        device, 
                                        optimizer,
                                        config.get('input_velocity_innovation_std')
                                        )

        training_errors.append(mean_error)

        model.eval()
        with torch.inference_mode():
            mean_error =_process_epoch_pass(model, 
                                            validation_data_loader, 
                                            device, 
                                            None)

            validation_errors.append(mean_error)
            # For periodic evaluation, save validation trajectories from last epoch
            if (
                epoch == config['epochs'] - 1
                and hasattr(validation_data_loader.dataset, "datasets")
            ):
                validation_trajectory_rollouts = _evaluate_full_trajectory_rollouts(
                    model,
                    validation_data_loader.dataset,
                    device,
                )

        if epoch % 2 == 0:
            print   ('epoch', epoch)

    return training_errors, validation_errors, validation_trajectory_rollouts

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a phase 1.0 position predictor.")
    parser.add_argument("--world-builder", default="simple-rolling")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # disabled for now since errors are note primarily due to that
    parser.add_argument("--input-velocity-innovation-std", type=float, default=0)  # DISABLE BY DEFAULT FOR NOW - used 3e-3 as a reasonable ON value
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
   # parser.add_argument("--output-dir", default="artifacts/phase1_0/run_001")
    return parser.parse_args()

def main() -> None:
    args = get_args()

    import genesis as gs

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        gs.init(backend=gs.gpu, seed=args.seed)
    else:
        gs.init(backend=gs.gpu)

    world_builder = load_world_builder(args.world_builder)

    trajectories, training_dataset, validation_dataset = create_trajectory_dataset(
        world_builder,
        args.seed,
        args.steps,
        args.trajectories,
        HISTORY_WINDOW,
        PREDICTION_HORIZON,
    )

    # Genesis sets PyTorch's default device to its backend during initialization.
    # Keep DataLoader indices and shuffle permutations on CPU.
    torch.set_default_device("cpu")
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PositionHistoryMLP(
        HISTORY_WINDOW,
        args.hidden_size,
        scene_feature_size=world_builder.scene_feature_size,
    ).to(device)

    config = {'lr': args.learning_rate,
              'epochs': args.epochs,
              'input_velocity_innovation_std': args.input_velocity_innovation_std,
              }

    # Later move into a dedicated class
    training_errors, validation_errors, validation_trajectory_rollouts = train_model(
        model,
        training_loader,
        validation_loader,
        config,
        device,
    )

    output_dir = Path('artifacts/runs')

    output_dir.mkdir(parents=True, exist_ok=True)
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

    trajectory_lengths = np.asarray([len(trajectory) for trajectory in trajectories])
    saved_positions = np.full(
        (len(trajectories), trajectory_lengths.max(), 3),
        np.nan,
        dtype=np.float32,
    )
    for trajectory_index, trajectory in enumerate(trajectories):
        saved_positions[trajectory_index, :len(trajectory)] = trajectory
    np.savez(
        output_dir / "trajectory.npz",
        positions=saved_positions,
        trajectory_lengths=trajectory_lengths,
        scene_features=np.stack(
            [
                trajectory_dataset.scene_features
                for trajectory_dataset in (
                    training_dataset.datasets + validation_dataset.datasets
                )
            ]
        ),
    )
    torch.save(
        {
            "model_type": "PositionHistoryMLP",
            "world_builder": args.world_builder,
            "history_length": HISTORY_WINDOW,
            "prediction_horizon": PREDICTION_HORIZON,
            "hidden_size": args.hidden_size,
            "scene_feature_size": model.scene_feature_size,
            "state_dict": model.state_dict(),
        },
        output_dir / "checkpoint.tar",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "training_error": training_errors,
                "validation_error": validation_errors,
            },
            indent=2,
        )
    )
    print(f"Saved phase 1.0 artifacts to {output_dir}")


if __name__ == "__main__":
    main()
