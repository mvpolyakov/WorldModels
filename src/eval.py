import genesis as gs
import argparse
import os
import time
import numpy as np
import torch

from sim.build_world import load_world_builder
from sim.dataclasses import add_scene_controls
from sim.utilities.sim_utils import (
    DisplacementErrorTracker,
    PredictionWorld,
)
from sim.utilities.viz_utils import save_displacement_error_chart
from world_models import DummyWorldModel, PositionHistoryMLP

def get_args():
    parser = argparse.ArgumentParser(description="Build and run GRI eval.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ghost-interval", type=int, default=10)
    parser.add_argument("--output-image", type=str, default="artifacts/ball_path.png")
    parser.add_argument("--output-error-chart", type=str, default="artifacts/displacement_error.png")

    parser.add_argument("--world", type=str, default="simple-rolling")
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

    if args.model == '':
        print('loading dummy random model')
        model = DummyWorldModel()
    else:
        print('loading ', args.model)
        checkpoint = torch.load(args.model, map_location=args.device)
        model = PositionHistoryMLP(
            history_length=checkpoint["history_length"],
            hidden_size=checkpoint["hidden_size"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(args.device)
        model.eval()

    # Create world

    world_builder = load_world_builder(args.world)

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

    add_scene_controls(world)

    # Set up prediction and trajectory recording

    initial_mobile_positions = np.stack(
        [
            mobile.get_pos().detach().cpu().numpy()
            for mobile in world.mobiles.values()
        ],
        axis=1,
    )
    predicted_world = PredictionWorld(initial_mobile_positions, model)
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

        if predicted_world.needs_warm_up():
            predicted_world.observe(actual_positions)
            if (step + 1) % args.ghost_interval == 0:
                trajectory_recorder.capture_observation(actual_positions)
            continue

        predicted_world.step()
        displacement_error.update(actual_positions, predicted_world.mobile_positions)

        if trajectory_recorder is not None and (step + 1) % args.ghost_interval == 0:
            trajectory_recorder.capture()

    print(
        f"Final cumulative displacement error after {len(displacement_error.values)} steps: "
        f"{displacement_error.values[-1]:.6f}"
    )
    error_chart_path = save_displacement_error_chart(
        displacement_error.values,
        args.output_error_chart,
    )
    print(f"Saved displacement error chart to {os.path.abspath(error_chart_path)}")

    if trajectory_recorder is not None:
        output_path = trajectory_recorder.save_image(args.output_image)
        print(f"Saved ball path image to {os.path.abspath(output_path)}")

    # save frames ??

    # eval script is specifically for a viewer-enabled sim
    print("Simulation finished. Close the viewer window to exit.")
    while world.scene.viewer.is_alive():  # pyright: ignore
        time.sleep(0.1)

if __name__ == "__main__":
    main()
