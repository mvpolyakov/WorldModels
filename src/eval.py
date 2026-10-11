import genesis as gs
import argparse
import os
import time
import numpy as np
import torch

from sim.build_world import load_world_builder
from sim.dataclasses import World, add_scene_controls
from sim.scene_features import (
    SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
    SURFACE_FEATURE_SCHEMA,
    find_closest_surface,
    unpack_surface_features,
)
from sim.utilities.sim_utils import (
    DisplacementErrorTracker,
    PredictionWorld,
    TrajectoryRecorder,
)
from sim.utilities.viz_utils import (
    draw_ground_coordinate_lines,
    save_displacement_error_chart,
)
from world_models import (
    DummyWorldModel,
    LatentStateModel,
    PositionHistoryMLP,
)

INITIAL_WARM_UP_LENGTH = 20


def _resolve_world_builder_name(
    requested_world: str | None,
    checkpoint: dict | None,
) -> str:
    """Select an explicit, checkpoint-recorded, or compatible default world."""
    if requested_world is not None:
        return requested_world
    if checkpoint is not None:
        return checkpoint.get("world_builder", "simple-rolling")
    return "opposing-ramps"


def _validate_scene_feature_size(model, world_scene_features, world_builder_name: str) -> None:
    """Reject a world whose scene features cannot be consumed by the model."""
    if not isinstance(model, PositionHistoryMLP):
        return
    actual_feature_size = world_scene_features.shape[1]
    if actual_feature_size != model.scene_feature_size:
        raise ValueError(
            f"World builder '{world_builder_name}' produces "
            f"{actual_feature_size} scene features, but this model expects "
            f"{model.scene_feature_size}."
        )


def _load_checkpoint_model(checkpoint, device):
    """Construct the recorded model class and load its parameter tensors.

    @param checkpoint: Checkpoint dictionary containing model_type, constructor
        dimensions, and state_dict.
    @param device: Torch device receiving the reconstructed model.

    @return: Model in evaluation mode with checkpoint parameters loaded.
    """
    if checkpoint.get("model_type") == "PositionHistoryMLP":
        model = PositionHistoryMLP(
            history_length=checkpoint["history_length"],
            hidden_size=checkpoint["hidden_size"],
            scene_feature_size=checkpoint.get(
                "scene_feature_size",
                SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
            ),
        )
    elif checkpoint.get("model_type") == "LatentStateModel":
        if checkpoint.get("scene_feature_schema") != SURFACE_FEATURE_SCHEMA:
            raise ValueError(
                "Latent evaluation requires a top_planes_v1 surface-context "
                "checkpoint. Older ramp-pose checkpoints require retraining."
            )
        initialization_gains = {
            name: checkpoint[name]
            for name in ("dynamics_output_weight_gain", "decoder_output_weight_gain")
            if name in checkpoint
        }
        model = LatentStateModel(
            latent_size=checkpoint["latent_size"],
            hidden_size=checkpoint["hidden_size"],
            warm_up_length=checkpoint["warm_up_length"],
            layer_norm_enabled=checkpoint.get("layer_norm_enabled", False),
            **initialization_gains,
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def _collect_actual_trajectory(
    world: World,
    step_count: int,
    warm_up_length: int,
) -> dict:
    """Run the simulation once and retain every actual position for replay.

    @param world: Simulation to advance while recording all mobile positions.
    @param step_count: Simulation steps to collect after recording frame zero.
    @param warm_up_length: Number of initial actual frames copied before prediction.

    @return: Recording containing actual positions shaped (T, 1, K, 3), where
        T is frame count, 1 is the single Genesis environment, and K is mobile
        count; frame indices; scene features; and the required warm-up length.
    """
    initial_mobile_positions = [
        np.stack(
            [
                mobile.get_pos().detach().cpu().numpy() # .get_pos() returns shape [1, 3]
                for mobile in world.mobiles.values()
            ],
            axis=1,
        )
    ]
    for _ in range(step_count):
        world.step()
        initial_mobile_positions.append(
            np.stack(
                [
                    mobile.get_pos().detach().cpu().numpy()
                    for mobile in world.mobiles.values()
                ],
                axis=1,
            )
        )

    return {
        "actual_positions": np.stack(initial_mobile_positions),
        "frame_indices": np.arange(step_count + 1),
        "scene_features": np.asarray(world.scene_features).copy(),
        "warm_up_length": warm_up_length,
    }


def _collect_surface_episode_metadata(
    actual_positions: np.ndarray,
    surface_features: np.ndarray,
    warm_up_length: int,
) -> dict:
    """Identify actual surface transitions and their restart observation windows.

    Use the same infinite-plane selection rule as the observation encoder. Keep
    every chronological transition occurrence, including repeated visits and
    transitions that lack a complete destination-only warm-up window.

    @param actual_positions: Recorded positions shaped (T, 1, K, 3), where T is
        frame count and K is mobile count. Phase 2 evaluation uses one ball.
    @param surface_features: Packed top_planes_v1 records shaped (1, 6*S + 1),
        where S is the number of surfaces.
    @param warm_up_length: Number of consecutive destination-surface observations
        required to restart the model.

    @return: Dictionary containing the surface schema, closest surface indices
        shaped (T,), and chronological transition records. Each record contains
        source surface, destination surface, transition frame, occurrence number,
        and either its complete observation window or an ineligible-window marker.
    """
    if actual_positions.ndim != 4 or actual_positions.shape[1:3] != (1, 1):
        raise ValueError(
            "Surface episode collection requires positions shaped (T, 1, 1, 3)."
        )
    if warm_up_length < 1:
        raise ValueError("warm_up_length must be at least one frame.")

    packed_surface_features = torch.as_tensor(
        surface_features,
        dtype=torch.float32,
    )
    surface_centers, surface_normals, _ = unpack_surface_features(
        packed_surface_features
    )
    frame_count = len(actual_positions)

    # compute closest surface for each frame
    closest_surface_indices = find_closest_surface(
        torch.as_tensor(actual_positions[:, 0, 0, :], dtype=torch.float32),
        surface_centers.expand(frame_count, -1, -1),
        surface_normals.expand(frame_count, -1, -1),
    ).cpu().numpy()

    surface_transitions = []
    for current_frame in range(1, frame_count):
        source_surface = int(closest_surface_indices[current_frame - 1])
        destination_surface = int(closest_surface_indices[current_frame])
        if source_surface == destination_surface:
            continue

        warmup_window_end = current_frame + warm_up_length

        # Check that the entire warmup period takes place on one surface. None retains
        # an incomplete or transient transition without allowing it to seed a latent-state restart.
        warmup_observation_window = None
        if (warmup_window_end <= frame_count and
            np.all(closest_surface_indices[current_frame:warmup_window_end] == destination_surface)
        ):
            warmup_observation_window = actual_positions[current_frame:warmup_window_end].copy()

        surface_transitions.append(
            {
                "source_surface": source_surface,
                "destination_surface": destination_surface,
                "transition_frame": current_frame,
                "occurrence": len(surface_transitions) + 1,
                "observation_window": warmup_observation_window,
            }
        )

    return {
        "scene_feature_schema": SURFACE_FEATURE_SCHEMA,
        "closest_surface_indices": closest_surface_indices,
        "surface_transitions": surface_transitions,
    }


def _replay_prediction(
    world: World,
    trajectory_recorder: TrajectoryRecorder,
    model,
    recording: dict,
    prediction_count: int,
    ghost_interval: int,
) -> dict:
    """Replay a model against positions collected before prediction.

    All models share observation warm-up, autonomous steps, recorded-actual
    playback, error measurement, ghost capture, and viewer refresh. Latent models
    additionally process predicted surface transfers after each autonomous step.

    @param world: Scene used for recorded playback, without further simulation.
    @param trajectory_recorder: Display recorded observations and predictions.
    @param model: World model to evaluate.
    @param recording: Collected actual frames, surface records, and transitions.
    @param prediction_count: Maximum autonomous steps after warm-up.
    @param ghost_interval: Frame spacing for trajectory snapshots.

    @return: Replay results containing cumulative XYZ displacement errors,
        prediction and segment counts, transfer timing records, and any stop
        frame and reason.
    """

    actual_positions = recording["actual_positions"]
    if isinstance(model, LatentStateModel):
        if recording.get("scene_feature_schema") != SURFACE_FEATURE_SCHEMA:
            raise ValueError(
                "Latent replay requires top_planes_v1 trajectory surface records."
            )
        scene_features = recording["surface_features"]
    else:
        scene_features = world.scene_features
    predicted_world = PredictionWorld(
        actual_positions[0],
        model,
        scene_features,
    )
    displacement_error = DisplacementErrorTracker()
    trajectory_recorder.set_up_predictions(predicted_world.mobile_positions)
    trajectory_recorder.capture_observation(actual_positions[0])
    final_actual_frame = len(actual_positions) - 1
    initial_warm_up_length = recording["warm_up_length"]
    prediction_count_completed = 0
    stop_reason = None
    stop_frame = None
    transfer_events = []
    segment_count = 1

    # skip any transitions that happen during the initial warmup period
    # this explicit handling is only needed at the beginning, not subsequent warmup periods
    if isinstance(model, LatentStateModel):
        next_transition_index = 0
        initial_anchor = min(initial_warm_up_length - 1, final_actual_frame)
        transitions = recording["surface_transitions"]
        while (
            next_transition_index < len(transitions)
            and transitions[next_transition_index]["transition_frame"] <= initial_anchor
        ):
            next_transition_index += 1
        surface_replay_state = {
            "previous_surface_index": int(
                recording["closest_surface_indices"][initial_anchor]
            ),
            "next_transition_index": next_transition_index,
        }

    frame_index = 1
    while (
        frame_index <= final_actual_frame
        and prediction_count_completed < prediction_count
    ):
        actual_frame_positions = actual_positions[frame_index]
        for mobile_index, mobile in enumerate(world.mobiles.values()):
            mobile.set_pos(actual_frame_positions[:, mobile_index, :])

        if (frame_index < initial_warm_up_length or
            predicted_world.needs_more_warm_up()
        ):
            predicted_world.observe(actual_frame_positions)
            if frame_index % ghost_interval == 0:
                trajectory_recorder.capture_observation(actual_frame_positions)
            world.scene.visualizer.update(force=True)
            frame_index += 1
            continue

        predicted_world.step()
        displacement_error.update(
            actual_frame_positions,
            predicted_world.mobile_positions,
        )
        prediction_count_completed += 1
        if frame_index % ghost_interval == 0:
            trajectory_recorder.capture()
        world.scene.visualizer.update(force=True)

        if isinstance(model, LatentStateModel):
            # model currently doesn't know about surface changes since the whole
            # eval is one autonomous rollout - if it switching surfaces look up when actual
            # rollout switched and relocate the mobile there and update surface
            surface_transfer_event = _detect_and_handle_surface_transfer(
                predicted_world,
                trajectory_recorder,
                recording,
                surface_replay_state,
            )
            if surface_transfer_event is not None:
                stop_reason = surface_transfer_event.get("stop_reason")
                if stop_reason is not None:
                    stop_frame = frame_index
                    break

                # this is needed for keeping overall frame_index on track (which tracks the rollout above)
                actual_transition_frame = surface_transfer_event["actual_transition_frame"]

                # just for printing out/recordkeeping at the end
                transfer_events.append(
                    {
                        "segment": segment_count,
                        "prediction_step": prediction_count_completed,
                        "predicted_transfer_frame": frame_index,
                        "actual_transition_frame": actual_transition_frame,
                        "frame_offset": frame_index - actual_transition_frame,
                    }
                )
                segment_count += 1
                frame_index = surface_transfer_event["actual_frame_anchor"]
        frame_index += 1

    if (
        stop_reason is None
        and prediction_count_completed < prediction_count
        and frame_index > final_actual_frame
    ):
        stop_frame = final_actual_frame
        stop_reason = "Recorded actual trajectory exhausted."

    return {
        "error_values": displacement_error.values,
        "prediction_count": prediction_count_completed,
        "segment_count": segment_count,
        "transfer_events": transfer_events,
        "stop_frame": stop_frame,
        "stop_reason": stop_reason,
    }


def _detect_and_handle_surface_transfer(
    predicted_world: PredictionWorld,
    trajectory_recorder: TrajectoryRecorder,
    recording: dict,
    replay_state: dict,
) -> dict | None:
    """Detect one predicted transfer and apply its matched observation restart.

    Compare the latest predicted surface with the preceding predicted surface. If
    it changed, match only the next chronological actual transition by source and
    destination. Retain the transfer prediction, reset the latent from the matched
    actual window, display those observations, break the prediction trail, and
    return the new actual-frame anchor. Return a stop event for mismatches or
    incomplete windows, and None when no predicted transfer occurred.

    The implementation's event state and latent reset belong in a focused helper
    alongside PredictionWorld. Surface geometry detects events only and is not
    supplied to autonomous dynamics.

    @param predicted_world: Latent prediction state after its latest autonomous step.
    @param trajectory_recorder: Displays the matched observation window and marks
        the next prediction as a new segment.
    @param recording: Actual positions, surface records, and transition metadata.
    @param replay_state: Mutable event state containing the preceding predicted
        surface and next chronological transition index.

    @return: None when prediction remains on its current surface. Otherwise, an
        event containing either a new actual-frame anchor or a stop reason.
    """
    predicted_positions = predicted_world.mobile_positions
    if predicted_positions.shape != (1, 1, 3):
        raise ValueError(
            "Surface-triggered replay currently requires one environment and one mobile."
        )

    surface_features = torch.as_tensor(
        recording["surface_features"],
        dtype=torch.float32,
    )
    surface_centers, surface_normals, _ = unpack_surface_features(surface_features)
    predicted_surface_index = int(
        find_closest_surface(
            torch.as_tensor(predicted_positions[:, 0, :], dtype=torch.float32),
            surface_centers,
            surface_normals,
        ).item()
    )

    previous_surface_index = replay_state["previous_surface_index"]
    if predicted_surface_index == previous_surface_index:
        replay_state["previous_surface_index"] = predicted_surface_index
        return None

    # Based on model's position, it is changing closest surface on this step

    transitions = recording["surface_transitions"]
    transition_index = replay_state["next_transition_index"]
    if transition_index >= len(transitions):
        return {
            "stop_reason": "No remaining actual surface transition matches the prediction.",
        }

    # what actually happened in the recorded scene next
    transition = transitions[transition_index]
    expected_pair = (
        transition["source_surface"],
        transition["destination_surface"],
    )

    # what the model expects to happen
    predicted_pair = (previous_surface_index, predicted_surface_index)

    if predicted_pair != expected_pair:
        return {
            "stop_reason": (
                f"Predicted transfer {predicted_pair} does not match next actual "
                f"transfer {expected_pair}."
            ),
        }

    replay_state["next_transition_index"] += 1
    warmup_observation_window = transition.get("observation_window")
    if warmup_observation_window is None:
        return {
            "stop_reason": "Matching actual transition has no complete observation window.",
        }

    predicted_world.reset_to_observation(warmup_observation_window[0])

    for window_frame, observed_positions in enumerate(warmup_observation_window):
        if window_frame > 0:
            predicted_world.observe(observed_positions) # For warming up the model
        for mobile_index, mobile in enumerate(
            trajectory_recorder.actual_mobiles.values()
        ):
            mobile.set_pos(observed_positions[:, mobile_index, :])
        trajectory_recorder.capture_observation(observed_positions)
        trajectory_recorder.scene.visualizer.update(force=True)

    replay_state["previous_surface_index"] = predicted_surface_index
    actual_frame_anchor = transition["transition_frame"] + len(warmup_observation_window) - 1

    return {
        "actual_frame_anchor": actual_frame_anchor,
        "actual_transition_frame": transition["transition_frame"],
        "stop_reason": None,
    }


def get_args():
    parser = argparse.ArgumentParser(description="Build and run GRI eval.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ghost-interval", type=int, default=10)
    parser.add_argument("--output-image", type=str, default="artifacts/ball_path.png")
    parser.add_argument("--output-error-chart", type=str, default="artifacts/displacement_error.png")

    parser.add_argument("--world", type=str, default="opposing-ramps")
    parser.add_argument("--randomize-scene", action="store_true")

    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")

    # later also add randomization options - with seed

    args = parser.parse_args()

    return args

def main() -> None:
    args = get_args()

    gs.init(backend=gs.gpu)  # pyright: ignore

    # Load model

    checkpoint = None
    if args.model == '':
        print('loading dummy random model')
        model = DummyWorldModel()
    else:
        print('loading ', args.model)
        checkpoint = torch.load(args.model, map_location=args.device)
        model = _load_checkpoint_model(checkpoint, args.device)

    # Create world

    world_builder_name = _resolve_world_builder_name(args.world, checkpoint)
    world_builder = load_world_builder(world_builder_name)
    initial_warm_up_length = max(
        INITIAL_WARM_UP_LENGTH, model.get_required_history()
    )

    random_seed = np.random.SeedSequence().entropy if args.randomize_scene else None
    collection_random_generator = (
        np.random.default_rng(random_seed) if random_seed is not None else None
    )
    replay_random_generator = (
        np.random.default_rng(random_seed) if random_seed is not None else None
    )

    collection_world, _ = world_builder.build(
        show_viewer=False,
        n_environments=1,
        ghost_count=0,
        random_generator=collection_random_generator,
    )
    _validate_scene_feature_size(
        model,
        collection_world.scene_features,
        world_builder_name,
    )
    print('\n\n collecting trajectory')
    time.sleep(1)
    recording = _collect_actual_trajectory(
        collection_world,
        step_count=args.steps,
        warm_up_length=initial_warm_up_length,
    )
    if isinstance(model, LatentStateModel):
        recording["surface_features"] = world_builder.get_surface_features(
            collection_world
        )
    collection_world.scene.destroy()
    del collection_world

    if isinstance(model, LatentStateModel):
        recording.update(
            _collect_surface_episode_metadata(
                recording["actual_positions"],
                recording["surface_features"],
                model.get_required_history(),
            )
        )

    surface_transitions = recording.get("surface_transitions", [])
    prediction_ghost_count = (
        1 + args.steps // args.ghost_interval + len(surface_transitions)
    )
    initial_warm_up_final_frame = min(initial_warm_up_length - 1, args.steps)
    initial_observation_ghost_count = 1 + initial_warm_up_final_frame // args.ghost_interval
    restart_observation_ghost_count = sum(
        len(transition["observation_window"])
        for transition in surface_transitions
        if transition["observation_window"] is not None
    )
    ghost_count = max(
        prediction_ghost_count,
        initial_observation_ghost_count + restart_observation_ghost_count,
    )

    world, trajectory_recorder = world_builder.build(
        show_viewer=True,
        n_environments=1,
        ghost_count=ghost_count,
        random_generator=replay_random_generator,
    )
    draw_ground_coordinate_lines(world.scene, world_builder.config)
    add_scene_controls(world)

    print('\n\n replaying predictions')
    time.sleep(1)

    replay_result = _replay_prediction(
        world,
        trajectory_recorder,
        model,
        recording,
        prediction_count=args.steps,
        ghost_interval=args.ghost_interval,
    )
    final_trajectory_image = trajectory_recorder.render_image()

    print("Simulation and replay finished. Close the viewer window to save artifacts and exit.")
#    while world.scene.viewer.is_alive():  # pyright: ignore
#        time.sleep(0.1)

    error_values = replay_result["error_values"]
    if error_values:
        print(
            f"Final cumulative XYZ displacement error after "
            f"{replay_result['prediction_count']} steps: {error_values[-1]:.6f}"
        )
    else:
        print("No prediction frames were available after warm-up.")
    if replay_result["stop_reason"] is not None:
        print(
            f"Prediction rollout stopped at actual frame "
            f"{replay_result['stop_frame']}: {replay_result['stop_reason']}"
        )
    if isinstance(model, LatentStateModel):
        print(
            "Segmented prediction with observation restarts: "
            f"{replay_result['segment_count']} segment(s)."
        )
        for transfer_event in replay_result["transfer_events"]:
            print(
                f"Segment {transfer_event['segment']} transfer: predicted at "
                f"actual frame {transfer_event['predicted_transfer_frame']}, "
                f"recorded at frame {transfer_event['actual_transition_frame']} "
                f"(offset {transfer_event['frame_offset']:+d})."
            )

    error_chart_path = save_displacement_error_chart(
        error_values,
        args.output_error_chart,
    )
    print(f"Saved displacement error chart to {os.path.abspath(error_chart_path)}")

    output_path = trajectory_recorder.save_image(
        args.output_image,
        final_trajectory_image,
    )
    print(f"Saved ball path image to {os.path.abspath(output_path)}")
    while world.scene.viewer.is_alive():  # pyright: ignore
        time.sleep(0.1)

if __name__ == "__main__":
    main()
