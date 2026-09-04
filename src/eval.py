'''
    process command args
    read configs
    load model
    build_scence based on configs
    Run N eval rollouts
        scene.build(n_envs=n_rollouts, env_spacing=(1.5, 1.5))
        collect data and generate artifacts
'''

import genesis as gs
import argparse
import os

from sim.build_world import build_world
from sim.scene_config import load_scene_config
from sim.dataclasses import add_scene_controls
from sim.utilities.sim_utils import DummyWorldModel, PredictionWorld

def get_args():
    parser = argparse.ArgumentParser(description="Build and run GRI eval.")
    parser.add_argument("-n", "--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ghost-interval", type=int, default=10)
    parser.add_argument("--output-image", type=str, default="artifacts/ball_path.png")

    parser.add_argument("--scene-config", type=str, default='base')

    #    parser.add_argument("--model", type=str, default="")
    # later also add randomization options - with seed

    args = parser.parse_args()

    return args

def main() -> None:
    args = get_args()

    gs.init(backend=gs.gpu)

    scene_config = load_scene_config(args.scene_config)


    ghost_count = (args.steps + args.ghost_interval - 1) // args.ghost_interval
    world, trajectory_recorder = build_world(
        scene_config,
        n_envs=args.n_envs,
        ghost_count=ghost_count,
    )

    add_scene_controls(world)

    # load model if any
    model = DummyWorldModel()

    predicted_world = PredictionWorld(scene_config.mobiles, model)

    if trajectory_recorder is not None:
        trajectory_recorder.set_up_predictions(predicted_world.mobile_positions)

    for step in range(args.steps):
        world.step()
        if predicted_world is not None:
            predicted_world.step()
        if trajectory_recorder is not None and (step + 1) % args.ghost_interval == 0:
            trajectory_recorder.capture()

    if trajectory_recorder is not None:
        output_path = trajectory_recorder.save_image(args.output_image)
        print(f"Saved ball path image to {os.path.abspath(output_path)}")

    # save frames ??

    print("Initial steps finished. Keeping viewer open (press Ctrl+C to exit)...")
    while True:
        # Keep stepping the world so the viewer stays responsive
        world.step()

if __name__ == "__main__":
    main()
