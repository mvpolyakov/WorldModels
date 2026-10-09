from collections.abc import Iterable

import torch


def _compute_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute the combined L2 norm of the available parameter gradients.

    @param parameters: Model parameters whose gradients will be measured. Parameters
        without a gradient are excluded.

    @return: The scalar L2 norm across all supplied gradients.
    """
    parameter_gradient_norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not parameter_gradient_norms:
        return 0.0
    return torch.stack(parameter_gradient_norms).norm(2).item()


def _compute_baseline_position_error_sums(
    observed_positions: torch.Tensor,
    padding_frame_mask: torch.Tensor,
    warmup_length: int,
    prediction_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score constant-position and constant-velocity forecasts like the model.

    Each baseline uses the same forecast starts, offsets, targets, and padding
    exclusions as the model's position loss.

    @param observed_positions: Actual XYZ positions shaped (B, T, 3).
    @param padding_frame_mask: Boolean tensor shaped (B, T); True excludes a frame.
    @param warmup_length: Observed frame count before the first forecast.
    @param prediction_horizon: Maximum number of autonomous forecast steps.

    @return: Scalar squared-error sums for constant position and constant velocity.
    """
    padded_frame_count = observed_positions.shape[1]
    constant_position_error_sum = observed_positions.new_zeros(())
    constant_velocity_error_sum = observed_positions.new_zeros(())
    forecast_start_count = max(padded_frame_count - warmup_length, 0)
    forecast_start_frames = torch.arange(
        warmup_length - 1,
        warmup_length - 1 + forecast_start_count,
        device=observed_positions.device,
    )
    maximum_forecast_offset = min(prediction_horizon, forecast_start_count)

    for forecast_offset in range(1, maximum_forecast_offset + 1):
        active_start_count = forecast_start_count - forecast_offset + 1
        active_start_frames = forecast_start_frames[:active_start_count]
        target_frames = active_start_frames + forecast_offset
        start_positions = observed_positions[:, active_start_frames]
        target_positions = observed_positions[:, target_frames]
        previous_positions = observed_positions[:, active_start_frames - 1]
        forecast_padding_mask = (
            padding_frame_mask[:, active_start_frames]
            | padding_frame_mask[:, target_frames]
        )
        scored_forecast_mask = (~forecast_padding_mask).unsqueeze(-1)

        constant_position_error_sum = constant_position_error_sum + torch.sum(
            (start_positions - target_positions).square() * scored_forecast_mask
        )
        constant_velocity_predictions = start_positions + forecast_offset * (
            start_positions - previous_positions
        )
        constant_velocity_error_sum = constant_velocity_error_sum + torch.sum(
            (constant_velocity_predictions - target_positions).square()
            * scored_forecast_mask
        )

    return constant_position_error_sum, constant_velocity_error_sum
