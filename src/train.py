import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sim.build_world import load_world_builder
from sim.scene_features import SURFACE_FEATURE_SCHEMA
from training.trajectories import (
    TrajectorySliceDataset,
    create_trajectory_sequence_dataset,
)
from training_profiling_utils import (
    _compute_baseline_position_error_sums,
    _compute_gradient_norm,
)
from world_models import LatentStateModel

# Min 2 frames for change detection since includes the frame from which we predict; three for acceleration. Couple extra for init stabilization
WARMUP_LENGTH = 5
PREDICTION_HORIZON = 10


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


def _compute_penetration_error_sum(
    model: LatentStateModel,
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    scene_features: torch.Tensor,
    padding_frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize predictions below the target frame's selected surface.

    The target position selects the surface, so a prediction cannot avoid the
    penalty by becoming closer to another plane. Negative target gaps establish
    the contact solver's allowed penetration for that frame.

    @param model: Model whose observation encoder owns surface selection.
    @param predicted_positions: Predicted XYZ positions shaped (B, 3).
    @param target_positions: Actual target XYZ positions shaped (B, 3).
    @param scene_features: Packed surface records shaped (B, 6*S + 1).
    @param padding_frame_mask: Boolean tensor shaped (B,); True excludes a row.

    @return: Scalar sum of squared excess penetration over non-padding rows.
    """
    target_surface_context = model.observation_encoder.compute_closest_surface_context(
        target_positions,
        scene_features,
    )
    # Target is used here because that's what's coming back from above line, just a computational trick
    target_surface_gap = target_surface_context[:, :1]
    selected_surface_normal = target_surface_context[:, 1:]
    predicted_surface_gap = target_surface_gap + torch.sum(
        (predicted_positions - target_positions) * selected_surface_normal,
        dim=1,
        keepdim=True,
    )
    minimum_allowed_gap = torch.minimum(
        target_surface_gap,
        torch.zeros_like(target_surface_gap),
    )
    excess_penetration = torch.relu(minimum_allowed_gap - predicted_surface_gap)
    return torch.sum(
        excess_penetration.square()
        * (~padding_frame_mask).unsqueeze(-1)
    )


def _compute_autonomous_rollout_error_sums(
    model: LatentStateModel,
    observation_states: torch.Tensor,
    observed_positions: torch.Tensor,
    scene_features: torch.Tensor,
    padding_frame_mask: torch.Tensor,
    compute_penetration_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Score all autonomous forecast starts in parallel at each forecast offset.

    B is the number of trajectory slices in the current DataLoader batch, T is
    the padded frame count per slice, W is WARMUP_LENGTH, L is latent width, and
    F is scene-feature width. Each slice still starts a forecast at every frame
    from W - 1 through T - 2. These T - W starts are stored in a second batch
    dimension: index 0 starts at frame W - 1, index 1 starts at frame W, and so
    on. States therefore begin with shape (B, T - W, L).

    With the current defaults B is normally 4, T is 64, and W is 5. The tensor
    therefore has 4 slice rows and 59 forecast-start columns at offset 1. It has
    58 columns at offset 2 and continues shrinking to 50 columns at offset 10.
    A final partial DataLoader batch can have fewer than 4 rows. A slice with
    fewer than 64 real frames still occupies one padded row; its padding mask
    excludes unavailable forecasts from the losses.

    The loop advances every start by one forecast offset at a time. Offset 1
    predicts the frame immediately after each start. Offset 2 advances those
    predicted states once more. Starts are removed from the end of the second
    dimension when the requested offset would pass frame T - 1. Each start
    retains its own recurrent state throughout.

    For a sequence ending at frame 7, starts at frames 4, 5, and 6 predict frames
    5, 6, and 7 together at offset 1. At offset 2, start 6 is dropped because
    frame 8 is unavailable; the carried states for starts 4 and 5 predict frames
    6 and 7 together.

    The two batch dimensions propagate as follows. Subrows within one offset are
     the B trajectory slices in the DataLoader batch, columns are forecast starts,
    and every cell contains L latent values. 
    ``s[b,t]`` is the observed state for slice b at frame t;
    ``p[b,a->t]`` is a forecast that started at frame a and has reached frame t.

    ::

                       forecast-start dimension R
                       start 4       start 5       start 6
        initial slice 0  s[0,4]        s[0,5]        s[0,6]
                slice 1  s[1,4]        s[1,5]        s[1,6]
                 ...      |              |              |
                          v              v              v
        offset 1 slice 0  p[0,4->5]     p[0,5->6]     p[0,6->7]
                 slice 1  p[1,4->5]     p[1,5->6]     p[1,6->7]
                 ...      |              |
                          v              v          column dropped
        offset 2 slice 0  p[0,4->6]     p[0,5->7]
                 slice 1  p[1,4->6]     p[1,5->7]
                 ...

    One model call advances every cell in an offset block. Vertical arrows retain
    recurrence within one slice and one forecast start; no state crosses a
    row or column.

    @param model: Latent model used for autonomous transitions and decoding.
    @param observation_states: Observation-informed states shaped (B, T, L).
    @param observed_positions: Actual XYZ positions shaped (B, T, 3).
    @param scene_features: Static packed surface features shaped (B, F).
    @param padding_frame_mask: Boolean tensor shaped (B, T); True excludes a frame.
    @param compute_penetration_loss: Whether to calculate penetration errors.

    @return: Position, penetration, and consistency error sums, followed by the
        number of non-padding forecast frames scored across all starts and offsets.
    """
    batch_size, padded_frame_count, _ = observed_positions.shape
    position_error_sum = observed_positions.new_zeros(())
    penetration_error_sum = observed_positions.new_zeros(())
    consistency_error_sum = observed_positions.new_zeros(())
    forecast_frame_count = torch.zeros(
        (),
        dtype=torch.long,
        device=observed_positions.device,
    )

    forecast_start_count = max(padded_frame_count - WARMUP_LENGTH, 0)
    forecast_start_frames = torch.arange(
        WARMUP_LENGTH - 1,
        WARMUP_LENGTH - 1 + forecast_start_count,
        device=observed_positions.device,
    )
    predicted_states = observation_states[:, forecast_start_frames]
    predicted_positions = observed_positions[:, forecast_start_frames]
    maximum_forecast_offset = min(
        PREDICTION_HORIZON,
        forecast_start_count,
    )

    for forecast_offset in range(1, maximum_forecast_offset + 1):
        # The latest start loses one available target at each new horizon.
        active_start_count = forecast_start_count - forecast_offset + 1
        active_start_frames = forecast_start_frames[:active_start_count]
        target_frames = active_start_frames + forecast_offset

        predicted_states = model.predict_state(
            predicted_states[:, :active_start_count]
        )
        predicted_positions = (
            predicted_positions[:, :active_start_count]
            + model.decode_state(predicted_states)
        )
        target_positions = observed_positions[:, target_frames]
        forecast_padding_mask = (
            padding_frame_mask[:, active_start_frames]
            | padding_frame_mask[:, target_frames]
        )
        scored_forecast_mask = (~forecast_padding_mask).unsqueeze(-1)

        position_error_sum = position_error_sum + torch.sum(
            (predicted_positions - target_positions) ** 2
            * scored_forecast_mask
        )
        if compute_penetration_loss:
            flat_forecast_count = batch_size * active_start_count
            repeated_scene_features = scene_features.repeat_interleave(
                active_start_count,
                dim=0,
            )
            penetration_error_sum = (
                penetration_error_sum
                + _compute_penetration_error_sum(
                    model,
                    predicted_positions.reshape(flat_forecast_count, 3),
                    target_positions.reshape(flat_forecast_count, 3),
                    repeated_scene_features,
                    forecast_padding_mask.reshape(flat_forecast_count),
                )
            )
        consistency_error_sum = (
            consistency_error_sum
            + _compute_consistency_error_sum(
                predicted_states,
                observation_states[:, target_frames],
                forecast_padding_mask,
            )
        )
        forecast_frame_count = forecast_frame_count + scored_forecast_mask.sum()

    return (
        position_error_sum,
        penetration_error_sum,
        consistency_error_sum,
        int(forecast_frame_count.item()),
    )


def _compute_batch_losses(
    model: LatentStateModel,
    observed_positions: torch.Tensor,
    scene_features: torch.Tensor,
    padding_frame_mask: torch.Tensor,
    compute_penetration_loss: bool,
):
    """Compute reconstruction and autonomous forecast losses for a padded batch.

    B is trajectory count, T is padded frame count, L is latent width, and F is
    scene-feature width. Each forecast starts after WARMUP_LENGTH observations
    and advances up to PREDICTION_HORIZON frames without observation updates.

    @param observed_positions: XYZ observations (B, T, 3), with finite padding.
    @param scene_features: Static scene features (B, F).
    @param padding_frame_mask: Boolean tensor (B, T); True marks trailing padding.
    @param compute_penetration_loss: Whether to calculate penetration errors.

    @return: Scalar tensors position_loss, its matched constant-position and
        constant-velocity baselines, penetration_loss, reconstruction_loss,
        consistency_loss, and latent_std; their integer aggregation counts and
        observed_frame_count. Forecast losses are zero when no frames remain
        after warm-up.
    """
    batch_size, padded_frame_count, position_size = observed_positions.shape

    # Observation branch first since the autonomous rollouts depend on it
    # prediction_horizon doesn't play a role here - observation updates computed once per frame

    observation_states = _compute_observation_states(
        model, observed_positions, scene_features, padding_frame_mask
    )
    observed_frame_count = int((~padding_frame_mask).sum().item())

    # Frame zero has no preceding observation and therefore no displacement target.
    consecutive_frame_pair_count = max(padded_frame_count - 1, 0)
    reconstructed_positions = observed_positions[:, :-1] + model.decode_state(
        observation_states[:, 1:].reshape(
            batch_size * consecutive_frame_pair_count,
            model.latent_size,
        )
    ).reshape(batch_size, consecutive_frame_pair_count, position_size)
    reconstruction_error_sum = torch.sum(
        (reconstructed_positions - observed_positions[:, 1:]) ** 2
        * (~padding_frame_mask[:, 1:]).unsqueeze(-1)
    )
    reconstruction_element_count = int(
        (~padding_frame_mask[:, 1:]).sum().item()
    ) * position_size

    (
        position_error_sum,
        penetration_error_sum,
        consistency_error_sum,
        forecast_frame_count,
    ) = _compute_autonomous_rollout_error_sums(
        model,
        observation_states,
        observed_positions,
        scene_features,
        padding_frame_mask,
        compute_penetration_loss,
    )
    (
        constant_position_error_sum,
        constant_velocity_error_sum,
    ) = _compute_baseline_position_error_sums(
        observed_positions,
        padding_frame_mask,
        WARMUP_LENGTH,
        PREDICTION_HORIZON,
    )

    # Average over scored scalar values, not padded frames or trajectory count.
    # A frame can be scored by several forecasts starting at different times.
    position_element_count = forecast_frame_count * position_size
    penetration_element_count = forecast_frame_count
    consistency_element_count = forecast_frame_count * model.latent_size
    observed_states = observation_states[~padding_frame_mask]
    return {
        "position_loss": position_error_sum / max(position_element_count, 1),
        "constant_position_loss": (
            constant_position_error_sum / max(position_element_count, 1)
        ),
        "constant_velocity_loss": (
            constant_velocity_error_sum / max(position_element_count, 1)
        ),
        "penetration_loss": penetration_error_sum / max(
            penetration_element_count,
            1,
        ),
        "reconstruction_loss": reconstruction_error_sum / max(reconstruction_element_count, 1),
        "consistency_loss": consistency_error_sum / max(consistency_element_count, 1),
        "latent_std": observed_states.std(dim=0, unbiased=False).mean(),
        "position_element_count": position_element_count,
        "penetration_element_count": penetration_element_count,
        "reconstruction_element_count": reconstruction_element_count,
        "consistency_element_count": consistency_element_count,
        "observed_frame_count": observed_frame_count,
    }


def _build_batch_metrics(
    losses: dict[str, torch.Tensor | int],
    total_loss: torch.Tensor,
    model: LatentStateModel,
    *,
    include_gradient_norms: bool,
) -> dict[str, float | int]:
    """Build one persisted batch record from losses and post-backward gradients.

    Loss records include each MSE loss, latent standard deviation, total loss, and
    the element counts used by epoch aggregation. Training records additionally
    include gradient norms for the full model and each component.

    @param losses: Batch loss tensors and their aggregation counts.
    @param total_loss: Weighted sum of MSE values used for backpropagation.
    @param model: Model whose component gradients may be recorded.
    @param include_gradient_norms: Whether to append post-backward gradient norms.

    @return: JSON-serializable metrics for one batch.
    """
    batch_metrics = {
        "position_loss": losses["position_loss"].item(),
        "constant_position_loss": losses["constant_position_loss"].item(),
        "constant_velocity_loss": losses["constant_velocity_loss"].item(),
        "penetration_loss": losses["penetration_loss"].item(),
        "reconstruction_loss": losses["reconstruction_loss"].item(),
        "consistency_loss": losses["consistency_loss"].item(),
        "latent_std": losses["latent_std"].item(),
        "total_loss": total_loss.item(),
        "position_element_count": losses["position_element_count"],
        "penetration_element_count": losses["penetration_element_count"],
        "reconstruction_element_count": losses["reconstruction_element_count"],
        "consistency_element_count": losses["consistency_element_count"],
        "observed_frame_count": losses["observed_frame_count"],
    }
    if include_gradient_norms:
        batch_metrics.update(
            {
                "gradient_norm": _compute_gradient_norm(model.parameters()),
                "state_updater_gradient_norm": _compute_gradient_norm(
                    model.state_updater.parameters()
                ),
                "dynamics_model_gradient_norm": _compute_gradient_norm(
                    model.dynamics_model.parameters()
                ),
                "latent_state_decoder_gradient_norm": _compute_gradient_norm(
                    model.latent_state_decoder.parameters()
                ),
            }
        )
    return batch_metrics


def _process_epoch_pass(
    model: LatentStateModel,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    position_weight: float,
    reconstruction_weight: float,
    consistency_weight: float,
    batch_metrics: list[dict[str, float | int]] | None = None,
    *,
    penetration_weight: float = 1.0,
) -> dict[str, float]:
    """Run one training or validation pass over trajectory batches.

    Supplying an optimizer enables gradient updates; None computes validation
    metrics only. MSE losses are weighted by their scored scalar counts; latent_std
    is a frame-weighted mean of batch statistics, not a global standard deviation.

    @param model: LatentStateModel being trained or evaluated.
    @param data_loader: Loader yielding padded trajectory batches and masks.
    @param device: Torch device receiving each batch.
    @param optimizer: Optimizer for training, or None for validation.
    @param position_weight: Multiplier for autonomous position-prediction MSE.
    @param penetration_weight: Multiplier for excess predicted penetration MSE.
    @param reconstruction_weight: Multiplier for observation reconstruction MSE.
    @param consistency_weight: Multiplier for detached-target latent MSE.
    @param batch_metrics: Optional destination for one diagnostic record per batch.

    @return: Dictionary of MSE losses, latent standard deviation, and weighted
        MSE total_loss.
    """
    metric_sums = {
        "position_loss": 0.0,
        "constant_position_loss": 0.0,
        "constant_velocity_loss": 0.0,
        "penetration_loss": 0.0,
        "reconstruction_loss": 0.0,
        "consistency_loss": 0.0,
        "latent_std": 0.0,
    }
    metric_counts = {
        "position_loss": 0,
        "constant_position_loss": 0,
        "constant_velocity_loss": 0,
        "penetration_loss": 0,
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
            compute_penetration_loss=penetration_weight > 0,
        )
        total_loss = (
            position_weight * losses["position_loss"]
            + penetration_weight * losses["penetration_loss"]
            + reconstruction_weight * losses["reconstruction_loss"]
            + consistency_weight * losses["consistency_loss"]
        )
        if not torch.isfinite(total_loss):
            pass_name = "Validation" if optimizer is None else "Training"
            raise RuntimeError(f"{pass_name} loss became non-finite.")
        if optimizer is not None:
            total_loss.backward()
        if batch_metrics is not None:
            batch_metrics.append(
                _build_batch_metrics(
                    losses,
                    total_loss,
                    model,
                    include_gradient_norms=optimizer is not None,
                )
            )
        if optimizer is not None:
            optimizer.step()

        counts = {
            "position_loss": losses["position_element_count"],
            "constant_position_loss": losses["position_element_count"],
            "constant_velocity_loss": losses["position_element_count"],
            "penetration_loss": losses["penetration_element_count"],
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
        + penetration_weight * metrics["penetration_loss"]
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
    current_position = observed_positions[:, -1]
    for _ in range(prediction_count):
        predicted_displacement, state = model(state)
        current_position = current_position + predicted_displacement
        predicted_positions.append(current_position)

    if len(predicted_positions) == 0:
        return observed_positions.new_empty((observed_positions.shape[0], 0, 3))

    return torch.stack(predicted_positions, dim=1)


def _compute_trajectory_baseline_losses(
    actual_positions: torch.Tensor,
    history_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ten-frame baseline MSE at each forecast start in a trajectory.

    A result at rollout index zero starts from the final warm-up frame. Starts
    without a complete PREDICTION_HORIZON future are left as NaN so every plotted
    point summarizes the same number of forecast offsets.

    @param actual_positions: Complete XYZ trajectory shaped (T, 3).
    @param history_length: Number of initial observations used before rollout.

    @return: Constant-position and constant-velocity MSE vectors, each shaped
        (T - history_length,). Trailing entries without ten targets are NaN.
    """
    rollout_length = len(actual_positions) - history_length
    constant_position_losses = actual_positions.new_full((rollout_length,), torch.nan)
    constant_velocity_losses = actual_positions.new_full((rollout_length,), torch.nan)
    forecast_offsets = torch.arange(
        1,
        PREDICTION_HORIZON + 1,
        device=actual_positions.device,
    )

    for rollout_frame, forecast_start_frame in enumerate(
        range(history_length - 1, len(actual_positions) - PREDICTION_HORIZON)
    ):
        start_position = actual_positions[forecast_start_frame]
        previous_position = actual_positions[forecast_start_frame - 1]
        target_positions = actual_positions[
            forecast_start_frame + forecast_offsets
        ]
        constant_position_losses[rollout_frame] = (
            target_positions - start_position
        ).square().mean()
        constant_velocity_predictions = start_position + torch.unsqueeze(
            forecast_offsets,
            -1,
        ) * (start_position - previous_position)
        constant_velocity_losses[rollout_frame] = (
            target_positions - constant_velocity_predictions
        ).square().mean()

    return constant_position_losses, constant_velocity_losses


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
        positions; model errors; and time-indexed ten-frame baseline losses.
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
        (
            constant_position_losses,
            constant_velocity_losses,
        ) = _compute_trajectory_baseline_losses(
            actual_positions,
            history_length,
        )

        trajectory_rollouts.append(
            {
                "actual_positions": actual_future_positions.cpu().numpy(),
                "predicted_positions": predicted_positions.cpu().numpy(),
                "position_errors": position_errors.cpu().numpy(),
                "velocity_errors": velocity_errors.cpu().numpy(),
                "velocity_error_innovations": velocity_error_innovations.cpu().numpy(),
                "constant_position_losses": constant_position_losses.cpu().numpy(),
                "constant_velocity_losses": constant_velocity_losses.cpu().numpy(),
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

    @return: Per-epoch training metrics, per-epoch validation metrics, final
        full-trajectory validation rollouts, and per-epoch batch diagnostics.
    """
    optimizer = torch.optim.Adam(model.parameters(), config["lr"])
    training_metrics = []
    validation_metrics = []
    training_batch_metrics = []
    validation_batch_metrics = []
    validation_trajectory_rollouts = []

    for epoch in range(config["epochs"]):
        train_data_loader.dataset.resample_slices(epoch)
        model.train()
        epoch_training_batch_metrics = []
        training_metrics.append(
            _process_epoch_pass(
                model,
                train_data_loader,
                device,
                optimizer,
                config["position_weight"],
                config["reconstruction_weight"],
                config["consistency_weight"],
                epoch_training_batch_metrics,
                penetration_weight=config.get("penetration_weight", 1.0),
            )
        )
        training_batch_metrics.append(epoch_training_batch_metrics)

        model.eval()
        with torch.inference_mode():
            epoch_validation_batch_metrics = []
            validation_metrics.append(
                _process_epoch_pass(
                    model,
                    validation_data_loader,
                    device,
                    None,
                    config["position_weight"],
                    config["reconstruction_weight"],
                    config["consistency_weight"],
                    epoch_validation_batch_metrics,
                    penetration_weight=config.get("penetration_weight", 1.0),
                )
            )
            validation_batch_metrics.append(epoch_validation_batch_metrics)
            if epoch == config["epochs"] - 1:
                validation_trajectory_rollouts = _compute_full_trajectory_rollouts(
                    model,
                    validation_data_loader.dataset.source_dataset,
                    device,
                )

        if epoch % 2 == 0:
            print("epoch", epoch)

    batch_metrics = {
        "training": training_batch_metrics,
        "validation": validation_batch_metrics,
    }
    return (
        training_metrics,
        validation_metrics,
        validation_trajectory_rollouts,
        batch_metrics,
    )

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
    parser.add_argument("--hidden-size", type=int, default=16)  # not dynamics model
    parser.add_argument("--latent-size", type=int, default=8) # width of latent state
    parser.add_argument(
        "--layer-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--penetration-weight", type=float, default=1.0)
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
        trailing_shape = validation_trajectory_rollouts[0][key].shape[1:]
        values = np.full(
            (
                len(validation_trajectory_rollouts),
                max_validation_rollout_length,
                *trailing_shape,
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

    if args.backend == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU backend was requested, but PyTorch cannot access CUDA. "
            "Refusing to fall back to CPU."
        )

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
    if args.backend == "gpu" and gs.backend == gs.cpu:
        raise RuntimeError(
            "GPU backend was requested, but Genesis initialized its CPU backend. "
            "Refusing to continue with CPU simulation."
        )

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
        hidden_size=args.hidden_size,
        warm_up_length=WARMUP_LENGTH,
        layer_norm_enabled=args.layer_norm,
    ).to(device)
    config = {
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "position_weight": args.position_weight,
        "penetration_weight": args.penetration_weight,
        "reconstruction_weight": args.reconstruction_weight,
        "consistency_weight": args.consistency_weight,
    }
    (
        training_metrics,
        validation_metrics,
        validation_trajectory_rollouts,
        batch_metrics,
    ) = train_model(
        model,
        training_loader,
        validation_loader,
        config,
        device,
    )
    checkpoint = {
        "model_type": "LatentStateModel",
        "world_builder": args.world_builder,
        "latent_size": model.latent_size,
        "hidden_size": model.hidden_size,
        "scene_feature_size": model.observation_encoder.scene_feature_size,
        "scene_feature_schema": SURFACE_FEATURE_SCHEMA,
        "warm_up_length": model.warm_up_length,
        "layer_norm_enabled": model.layer_norm_enabled,
        "dynamics_output_weight_gain": model.dynamics_output_weight_gain,
        "decoder_output_weight_gain": model.decoder_output_weight_gain,
        "prediction_horizon": PREDICTION_HORIZON,
        "state_dict": model.state_dict(),
        "training_metadata": {
            "prediction_horizon": PREDICTION_HORIZON,
            "position_weight": args.position_weight,
            "penetration_weight": args.penetration_weight,
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
        "training_batches": batch_metrics["training"],
        "validation_batches": batch_metrics["validation"],
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
        scene_feature_schema=SURFACE_FEATURE_SCHEMA,
    )
    torch.save(checkpoint, output_dir / "checkpoint.tar")
    (output_dir / "metrics.json").write_text(
        json.dumps(saved_metrics, indent=2)
    )
    print(f"Saved training artifacts to {output_dir}")


if __name__ == "__main__":
    main()
