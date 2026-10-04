import genesis as gs
import argparse
import os
import time
import numpy as np
import torch

from sim.build_world import load_world_builder
from sim.dataclasses import add_scene_controls
from sim.scene_features import SIMPLE_ROLLING_SCENE_FEATURE_SIZE
from sim.utilities.sim_utils import (
    DisplacementErrorTracker,
    PredictionWorld,
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
    if not isinstance(model, (LatentStateModel, PositionHistoryMLP)):
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
        model = LatentStateModel(
            latent_size=checkpoint["latent_size"],
            observation_embedding_size=checkpoint.get(
                "observation_embedding_size", checkpoint["latent_size"]
            ),
            hidden_size=checkpoint["hidden_size"],
            scene_feature_size=checkpoint.get(
                "scene_feature_size",
                SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
            ),
            warm_up_length=checkpoint["warm_up_length"],
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model

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

    random_generator = np.random.default_rng(None) if args.randomize_scene else None

    world, trajectory_recorder = world_builder.build(
        show_viewer=True,
        n_environments=1,
        ghost_count=ghost_count,
        random_generator=random_generator,
    )
    _validate_scene_feature_size(
        model,
        world.scene_features,
        world_builder_name,
    )

    draw_ground_coordinate_lines(world.scene, world_builder.config)

    add_scene_controls(world)

    # Set up prediction and trajectory recording

    initial_mobile_positions = np.stack(
        [
            mobile.get_pos().detach().cpu().numpy()
            for mobile in world.mobiles.values()
        ],
        axis=1,
    )
    predicted_world = PredictionWorld(
        initial_mobile_positions,
        model,
        world.scene_features,
    )
    displacement_error = DisplacementErrorTracker()

    # for now everything below (in this file) assumes predicted world exists

    trajectory_recorder.set_up_predictions(predicted_world.mobile_positions)
    trajectory_recorder.capture_observation(initial_mobile_positions)

    for step in range(args.steps):
        world.step()

        actual_positions = np.stack(
            [
                mobile.get_pos().detach().cpu().numpy()
                for mobile in world.mobiles.values()
            ],
            axis=1,
        )

        # to check scale of inputs - true a few x positions are on the order of ~1, 
        # while others closer to .1, but not crazy
        if False: #step < 10:
            print(step,
                np.array2string(
                    actual_positions,
                    formatter={"float_kind": lambda value: f"{value:.2f}"},
                ),
                np.array2string(
                    world.scene_features,
                    formatter={"float_kind": lambda value: f"{value:.2f}"},
                ),
            )

        if predicted_world.needs_more_warm_up():
            predicted_world.observe(actual_positions)
            if (step + 1) % args.ghost_interval == 0:
                trajectory_recorder.capture_observation(actual_positions)
            continue

        predicted_world.step()
        displacement_error.update(actual_positions, predicted_world.mobile_positions)

        if trajectory_recorder is not None and (step + 1) % args.ghost_interval == 0:
            trajectory_recorder.capture()

    final_trajectory_image = (
        trajectory_recorder.render_image()
        if trajectory_recorder is not None
        else None
    )

    # eval script is specifically for a viewer-enabled sim
    print("Simulation finished. Close the viewer window to save artifacts and exit.")
    while world.scene.viewer.is_alive():  # pyright: ignore
        time.sleep(0.1)

    print(
        f"Final cumulative XY displacement error after "
        f"{len(displacement_error.values)} steps: "
        f"{displacement_error.values[-1]:.6f}"
    )
    error_chart_path = save_displacement_error_chart(
        displacement_error.values,
        args.output_error_chart,
    )
    print(f"Saved displacement error chart to {os.path.abspath(error_chart_path)}")

    if trajectory_recorder is not None:
        output_path = trajectory_recorder.save_image(
            args.output_image,
            final_trajectory_image,
        )
        print(f"Saved ball path image to {os.path.abspath(output_path)}")

    # save frames ??

if __name__ == "__main__":
    main()
