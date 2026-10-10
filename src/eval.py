import genesis as gs
import argparse
import os
import time
import numpy as np
import torch

from sim.build_world import load_world_builder
from sim.dataclasses import World, add_scene_controls
from sim.scene_features import SIMPLE_ROLLING_SCENE_FEATURE_SIZE, SURFACE_FEATURE_SCHEMA
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
    @param surface_features: Static top_planes_v1 records, shape (1, 6*S + 1).
    @param step_count: Simulation steps to collect after recording frame zero.
    @param warm_up_length: Observation count from model.get_required_history().

    @return: Recording containing actual positions shaped (T, 1, K, 3), frame
        indices, static surface records, and the required warm-up length. Surface
        indices and transitions are added with latent-state replay handling.
        T is length of trajectory, 1 is genisis's environments index - here = 1
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
        "surface_features": np.asarray(world.scene_features).copy(),
        "warm_up_length": warm_up_length,
    }


def _replay_prediction(
    world: World,
    trajectory_recorder: TrajectoryRecorder,
    model,
    recording: dict,
    prediction_count: int,
    ghost_interval: int,
) -> dict:
    """Replay an older model against positions collected before prediction.

    DummyWorldModel and PositionHistoryMLP retain their existing uninterrupted
    warm-up and prediction behavior. Latent-state surface matching and restarts
    can't be done this way.

    @param world: Scene used for recorded playback, without further simulation.
    @param trajectory_recorder: Display recorded observations and predictions.
    @param model: Dummy or position-history model to evaluate.
    @param recording: Collected actual frames, surface records, and transitions.
    @param prediction_count: Maximum autonomous steps after warm-up.
    @param ghost_interval: Frame spacing for trajectory snapshots.

    @return: Replay results containing cumulative displacement errors and the
        number of predictions compared with recorded actual positions.
    """

    actual_positions = recording["actual_positions"]
    predicted_world = PredictionWorld(
        actual_positions[0],
        model,
        world.scene_features,
    )
    displacement_error = DisplacementErrorTracker()
    trajectory_recorder.set_up_predictions(predicted_world.mobile_positions)
    trajectory_recorder.capture_observation(actual_positions[0])

    prediction_limit = min(
        len(actual_positions) - 1,
        model.get_required_history() - 1 + prediction_count,
    )
    prediction_count_completed = 0
    for frame_index in range(1, prediction_limit + 1):
        actual_frame_positions = actual_positions[frame_index]
        for mobile_index, mobile in enumerate(world.mobiles.values()):
            mobile.set_pos(actual_frame_positions[:, mobile_index, :])

        if predicted_world.needs_more_warm_up():
            predicted_world.observe(actual_frame_positions)
            if frame_index % ghost_interval == 0:
                trajectory_recorder.capture_observation(actual_frame_positions)
            world.scene.visualizer.update(force=True)
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

    return {
        "error_values": displacement_error.values,
        "prediction_count": prediction_count_completed,
        "has_observation_restarts": False,
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

    ghost_count = max(
        1 + args.steps // args.ghost_interval,
        model.get_required_history(),
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
        warm_up_length=model.get_required_history(),
    )
    collection_world.scene.destroy()
    del collection_world

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
    while world.scene.viewer.is_alive():  # pyright: ignore
        time.sleep(0.1)

    error_values = replay_result["error_values"]
    if error_values:
        print(
            f"Final cumulative XY displacement error after "
            f"{replay_result['prediction_count']} steps: {error_values[-1]:.6f}"
        )
    else:
        print("No prediction frames were available after warm-up.")

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

if __name__ == "__main__":
    main()
