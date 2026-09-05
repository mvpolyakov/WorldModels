import genesis as gs
import argparse
import os
import time
import numpy as np

from sim.build_world import build_world
from sim.scene_config import load_scene_config
from sim.dataclasses import add_scene_controls
from sim.utilities.sim_utils import (
    DisplacementErrorTracker,
    PredictionWorld,
)
from sim.utilities.viz_utils import save_displacement_error_chart
from world_models import DummyWorldModel

def get_args():
    parser = argparse.ArgumentParser(description="Build and run GRI eval.")
    parser.add_argument("-n", "--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ghost-interval", type=int, default=10)
    parser.add_argument("--output-image", type=str, default="artifacts/ball_path.png")
    parser.add_argument("--output-error-chart", type=str, default="artifacts/displacement_error.png")

    parser.add_argument("--scene-config", type=str, default='base')

    #    parser.add_argument("--model", type=str, default="")
    # later also add randomization options - with seed

    args = parser.parse_args()

    return args

def main() -> None:
    args = get_args()

    gs.init(backend=gs.gpu)  # pyright: ignore

    scene_config = load_scene_config(args.scene_config)


    ghost_count = (args.steps + args.ghost_interval - 1) // args.ghost_interval
    world, trajectory_recorder = build_world(
        scene_config,
        show_viewer=True,
        n_envs=args.n_envs,
        ghost_count=ghost_count,
    )

    add_scene_controls(world)

    # load model if any
    model = DummyWorldModel()

    predicted_world = PredictionWorld(scene_config.mobiles, model, n_envs=args.n_envs)
    displacement_error = DisplacementErrorTracker()

    # for now everything below (in this file) assumes predicted world exists

    if trajectory_recorder is not None:
        trajectory_recorder.set_up_predictions(predicted_world.mobile_positions)

    for step in range(args.steps):
        world.step()
        predicted_world.step()

        actual_positions = np.stack(
            [
                mobile.get_pos().detach().cpu().numpy()
                for mobile in world.mobiles.values()
            ],
            axis=1,
        )

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
