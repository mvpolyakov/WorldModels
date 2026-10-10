import argparse
from collections.abc import Callable, Iterable
from copy import deepcopy
import json
import math

import torch

from world_models import LatentStateModel


def _compute_rollout_drift_ratio(
    model: LatentStateModel,
    start_states: torch.Tensor,
    prediction_horizon: int,
    state_rms: float,
) -> float:
    """Measure maximum relative RMS latent drift from sampled starts (N, L).

    N is the number of forecast starts and L is latent width. Returns infinity
    when any rollout measurement is nonfinite.
    """
    predicted_states = start_states
    maximum_drift_ratio = 0.0
    for _ in range(prediction_horizon):
        predicted_states = model.predict_state(predicted_states)
        drift_ratio = (predicted_states - start_states).square().mean().sqrt().item() / state_rms
        if not math.isfinite(drift_ratio):
            return math.inf
        maximum_drift_ratio = max(maximum_drift_ratio, drift_ratio)
    return maximum_drift_ratio


@torch.no_grad()
def _compute_initialization_gains_for_model(
    fresh_model: LatentStateModel,
    observed_positions: torch.Tensor,
    scene_features: torch.Tensor,
    padding_frame_mask: torch.Tensor,
    prediction_horizon: int,
    *,
    rollout_drift_budget: float = 0.1,
    decoder_target_ratio: float = 1.0,
) -> dict[str, float]:
    """Measure suggested output gains for one fresh model initialization.

    Operates on a copy; weights and observation history are untouched. Use an
    untrained model with zero output biases. Calibration holds sampled
    observation-updated states fixed while rescaling the final weights.

    B is trajectory count, T is padded frame count, F is scene-feature width.
    Inputs must use the model's device and floating dtype; padding is trailing.

    @param observed_positions: XYZ positions (B, T, 3), including warm-up.
    @param scene_features: Static scene features (B, F).
    @param padding_frame_mask: Boolean (B, T); True excludes a padded frame.
    @param prediction_horizon: Number of repeated dynamics steps to check.
    @param rollout_drift_budget: Maximum RMS latent change / starting RMS across
        the rollout. The initial per-step target is this budget divided by horizon,
        since shared dynamics updates can accumulate coherently.
    @param decoder_target_ratio: Desired decoded RMS / observed displacement RMS.

    @return: Suggested constructor gains and measured RMS/ratio diagnostics before
        and after rescaling. Raises ValueError for absent transitions or zero/nonfinite
        signal scales; static data cannot determine a nonzero decoder gain.
    """
    if prediction_horizon < 1 or rollout_drift_budget <= 0 or decoder_target_ratio <= 0:
        raise ValueError("Horizon and calibration targets must be positive.")
    model = deepcopy(fresh_model).eval()
    for output_layer in (model.dynamics_model[-1], model.latent_state_decoder[-1]):
        if torch.count_nonzero(output_layer.bias):
            raise ValueError("Calibration requires a fresh model with zero output biases.")

    current_state = model.initialize_state(observed_positions.shape[0])
    forecast_states = []
    target_displacements = []
    for frame_index in range(observed_positions.shape[1] - 1):
        updated_state = model.encode_predict_update_helper(
            current_state,
            observed_positions[:, frame_index],
            scene_features,
            advance=frame_index > 0,
        )
        current_state = torch.where(
            padding_frame_mask[:, frame_index].unsqueeze(-1),
            current_state,
            updated_state,
        )
        if frame_index < model.warm_up_length - 1:
            continue
        observed_pair_mask = ~(
            padding_frame_mask[:, frame_index] | padding_frame_mask[:, frame_index + 1]
        )
        forecast_states.append(current_state[observed_pair_mask])
        target_displacements.append(
            (observed_positions[:, frame_index + 1] - observed_positions[:, frame_index])[
                observed_pair_mask
            ]
        )
    if not forecast_states or sum(len(states) for states in forecast_states) == 0:
        raise ValueError("No observed transitions remain after warm-up.")
    # Flatten the observed trajectory/start pairs; padded entries never enter RMS.
    start_states = torch.cat(forecast_states)
    displacements = torch.cat(target_displacements)
    state_rms = start_states.square().mean().sqrt().item()
    displacement_rms = displacements.square().mean().sqrt().item()
    residual_rms = model.dynamics_model(start_states).square().mean().sqrt().item()
    decoder_rms_before = model.decode_state(
        model.predict_state(start_states)
    ).square().mean().sqrt().item()
    scales = (state_rms, displacement_rms, residual_rms, decoder_rms_before)
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Calibration needs finite, nonzero latent, motion, and output RMS.")

    drift_ratio_before = _compute_rollout_drift_ratio(
        model, start_states, prediction_horizon, state_rms,
    )
    dynamics_multiplier = rollout_drift_budget / prediction_horizon * state_rms / residual_rms
    model.dynamics_model[-1].weight.mul_(dynamics_multiplier)
    # Check actual repeated transitions, not just a one-step linear approximation.
    for _ in range(24):
        maximum_drift_ratio = _compute_rollout_drift_ratio(
            model, start_states, prediction_horizon, state_rms,
        )
        if maximum_drift_ratio <= rollout_drift_budget:
            break
        model.dynamics_model[-1].weight.mul_(0.5)
        dynamics_multiplier *= 0.5
    else:
        raise ValueError("Could not meet the rollout drift budget by reducing dynamics gain.")

    decoded_rms = model.decode_state(
        model.predict_state(start_states)
    ).square().mean().sqrt().item()
    if not math.isfinite(decoded_rms) or decoded_rms <= 0:
        raise ValueError("Decoder has no finite nonzero signal after dynamics calibration.")
    decoder_multiplier = decoder_target_ratio * displacement_rms / decoded_rms
    model.latent_state_decoder[-1].weight.mul_(decoder_multiplier)
    return {
        "dynamics_output_weight_gain": fresh_model.dynamics_output_weight_gain * dynamics_multiplier,
        "decoder_output_weight_gain": fresh_model.decoder_output_weight_gain * decoder_multiplier,
        "observed_displacement_rms": displacement_rms,
        "latent_state_rms": state_rms,
        "residual_ratio_before": residual_rms / state_rms,
        "residual_ratio_after": model.dynamics_model(start_states).square().mean().sqrt().item() / state_rms,
        "rollout_drift_ratio_before": drift_ratio_before,
        "rollout_drift_ratio_after": maximum_drift_ratio,
        "rollout_drift_budget": rollout_drift_budget,
        "decoder_ratio_before": decoder_rms_before / displacement_rms,
        "decoder_ratio_after": model.decode_state(
            model.predict_state(start_states)
        ).square().mean().sqrt().item() / displacement_rms,
    }


def compute_initialization_gains(
    model_factory: Callable[[], LatentStateModel],
    observed_positions: torch.Tensor,
    scene_features: torch.Tensor,
    padding_frame_mask: torch.Tensor,
    prediction_horizon: int,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    rollout_drift_budget: float = 0.1,
    decoder_target_ratio: float = 1.0,
) -> dict[str, object]:
    """Recommend output gains from several fresh random initializations.

    The model factory must construct the exact architecture intended for training
    on the same device and dtype as the samples. The median is the recommendation
    because one unusually quiet or large random output is less influential than
    it is in the mean. Per-seed reports and means remain available for judging
    spread. CPU and CUDA random-number states are restored before returning.

    @param model_factory: Zero-argument constructor for a fresh untrained model.
    @param observed_positions: XYZ positions shaped (B, T, 3), including warm-up.
    @param scene_features: Static scene features shaped (B, F).
    @param padding_frame_mask: Boolean tensor shaped (B, T); True marks padding.
    @param prediction_horizon: Number of repeated dynamics steps to check.
    @param seeds: Distinct initialization seeds; defaults to five fixed seeds.
    @param rollout_drift_budget: Maximum relative RMS latent drift over the rollout.
    @param decoder_target_ratio: Desired decoded RMS / observed displacement RMS.

    @return: Recommended median gains, mean gains, sample standard deviations,
        seed count, and a list of the complete per-seed diagnostic reports.
    """
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("At least one initialization seed is required.")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("Initialization seeds must be distinct.")

    cuda_devices = []
    if observed_positions.device.type == "cuda":
        cuda_devices.append(observed_positions.device.index or torch.cuda.current_device())
    per_seed_reports = []
    with torch.random.fork_rng(devices=cuda_devices):
        for seed in seed_values:
            torch.manual_seed(seed)
            model = model_factory()
            report = _compute_initialization_gains_for_model(
                model,
                observed_positions,
                scene_features,
                padding_frame_mask,
                prediction_horizon,
                rollout_drift_budget=rollout_drift_budget,
                decoder_target_ratio=decoder_target_ratio,
            )
            per_seed_reports.append({"seed": seed, **report})

    dynamics_gains = torch.tensor(
        [report["dynamics_output_weight_gain"] for report in per_seed_reports]
    )
    decoder_gains = torch.tensor(
        [report["decoder_output_weight_gain"] for report in per_seed_reports]
    )
    return {
        "dynamics_output_weight_gain": dynamics_gains.median().item(),
        "decoder_output_weight_gain": decoder_gains.median().item(),
        "dynamics_output_weight_gain_mean": dynamics_gains.mean().item(),
        "decoder_output_weight_gain_mean": decoder_gains.mean().item(),
        "dynamics_output_weight_gain_std": dynamics_gains.std(unbiased=False).item(),
        "decoder_output_weight_gain_std": decoder_gains.std(unbiased=False).item(),
        "seed_count": len(seed_values),
        "per_seed": per_seed_reports,
    }


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


def _run_compute_init_gains() -> dict[str, object]:
    """Collect data with training defaults and compute initialization gains."""
    import genesis as gs

    from sim.build_world import load_world_builder
    from train import PREDICTION_HORIZON, WARMUP_LENGTH, get_args
    from training.trajectories import create_trajectory_sequence_dataset

    training_defaults = get_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genesis_backend = gs.gpu if device.type == "cuda" else gs.cpu
    if training_defaults.seed == 0:
        gs.init(backend=genesis_backend)
    else:
        gs.init(backend=genesis_backend, seed=training_defaults.seed)

    world_builder = load_world_builder(training_defaults.world_builder)
    _, training_dataset, _ = create_trajectory_sequence_dataset(
        world_builder,
        training_defaults.seed,
        training_defaults.steps,
        training_defaults.trajectories,
    )
    # Genesis changes PyTorch's default device; datasets and loader bookkeeping
    # remain on CPU while the complete calibration batch moves explicitly.
    torch.set_default_device("cpu")
    observed_positions = training_dataset.padded_positions.to(device)
    scene_features = training_dataset.scene_feature_tensor.to(device)
    padding_frame_mask = training_dataset.padding_frame_mask.to(device)

    def build_fresh_model() -> LatentStateModel:
        """Construct the architecture selected by the current training defaults."""
        return LatentStateModel(
            latent_size=training_defaults.latent_size,
            hidden_size=training_defaults.hidden_size,
            warm_up_length=WARMUP_LENGTH,
            layer_norm_enabled=training_defaults.layer_norm,
        ).to(device)

    return compute_initialization_gains(
        build_fresh_model,
        observed_positions,
        scene_features,
        padding_frame_mask,
        PREDICTION_HORIZON,
    )


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


def main(argv: list[str] | None = None) -> None:
    """
    This should be run whenever architecture or training scenario changes significantly
    Then compare running with prior values to running with the ones printed by this script 

    Run the requested training diagnostic and print its JSON report.
    
    """
    parser = argparse.ArgumentParser(description="Run training profiling utilities.")
    parser.add_argument("action", choices=("compute_init_gains",))
    args = parser.parse_args(argv)

    if args.action == "compute_init_gains":
        print(json.dumps(_run_compute_init_gains(), indent=2))


if __name__ == "__main__":
    main()
