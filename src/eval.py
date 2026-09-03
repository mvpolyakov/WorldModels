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

from sim.build_world import build_world
from sim.scene_config import load_scene_config
from sim.dataclasses import add_scene_controls

def get_args():
    parser = argparse.ArgumentParser(description="Build and run GRI eval.")
    parser.add_argument("-n", "--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)

    # load available options and show here on demand
    parser.add_argument("--scene-config", type=str, default='base')
    #    parser.add_argument("--model", type=str, default="")
    # later also add randomization options - with seed

    args = parser.parse_args()

    return args

def main() -> None:
    args = get_args()

    gs.init(backend=gs.gpu)

    scene_config = load_scene_config(args.scene_config)

    world = build_world(scene_config, n_envs=args.n_envs)

    add_scene_controls(world)

    # load model if any

    for _ in range(args.steps):
        world.step()

    # save frames ??

    print("Initial steps finished. Keeping viewer open (press Ctrl+C to exit)...")
    while True:
        # Keep stepping the world so the viewer stays responsive
        world.step()

if __name__ == "__main__":
    main()