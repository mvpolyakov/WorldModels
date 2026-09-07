from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


from sim.scene_config import SceneConfig


def create_randomized_rolling_scene(
    scene_config: SceneConfig,
    random_generator: np.random.Generator,
) -> SceneConfig:
    """Copy and randomize the ramp, ball, and their contact friction."""
    
    randomized_scene_config = copy.deepcopy(scene_config)
    ramp_config = next(obj for obj in randomized_scene_config.landscape if obj.name == "ramp")
    ground_config = next(obj for obj in randomized_scene_config.landscape if obj.name == "ground")
    ball_config = next(obj for obj in randomized_scene_config.mobiles if obj.name == "ball")

    friction = random_generator.uniform(0.1, 1.0)
    ramp_angle_degrees = random_generator.uniform(10.0, 30.0)
    ramp_angle_radians = math.radians(ramp_angle_degrees)
    ramp_direction_degrees = random_generator.uniform(-180.0, 180.0)
    ramp_direction_radians = math.radians(ramp_direction_degrees)
    distance_up_ramp_fraction = random_generator.uniform(0.55, 0.9)

    ramp_length, _, ramp_thickness = ramp_config.morph_params["size"]
    ramp_center_x, ramp_center_y, _ = ramp_config.morph_params["pos"]
    ground_center_z = ground_config.morph_params["pos"][2]
    ground_thickness = ground_config.morph_params["size"][2]
    ground_surface_z = ground_center_z + ground_thickness / 2.0
    ramp_half_length = ramp_length / 2.0
    ramp_half_thickness = ramp_thickness / 2.0

    ramp_center_z = (
        ground_surface_z
        + ramp_half_length * math.sin(ramp_angle_radians)
        + ramp_half_thickness * math.cos(ramp_angle_radians)
    )
    ramp_euler = ramp_config.morph_params.get("euler", (0.0, 0.0, 0.0))
    ramp_config.morph_params["pos"] = (ramp_center_x, ramp_center_y, ramp_center_z)
    ramp_config.morph_params["euler"] = (
        ramp_euler[0],
        -ramp_angle_degrees,
        ramp_direction_degrees,
    )

    distance_up_ramp = ramp_length * distance_up_ramp_fraction
    ball_radius = ball_config.morph_params["radius"]
    ball_clearance = 0.01
    distance_above_ramp_center = ramp_half_thickness + ball_radius + ball_clearance
    ball_horizontal_offset = (
        (ramp_half_length - distance_up_ramp) * math.cos(ramp_angle_radians)
        + distance_above_ramp_center * math.sin(ramp_angle_radians)
    )
    ball_x = ramp_center_x + ball_horizontal_offset * math.cos(ramp_direction_radians)
    ball_y = ramp_center_y + ball_horizontal_offset * math.sin(ramp_direction_radians)
    ball_z = (
        ramp_center_z
        + (distance_up_ramp - ramp_half_length) * math.sin(ramp_angle_radians)
        + distance_above_ramp_center * math.cos(ramp_angle_radians)
    )
    ball_config.morph_params["pos"] = (ball_x, ball_y, ball_z)

    ramp_config.friction = friction
    ball_config.friction = friction
    return randomized_scene_config


class SimulationTrajectoryCollector:
    """Collect one ball-position trajectory from a Genesis world."""

    def __init__(self, scene_config: SceneConfig):
        if len(scene_config.mobiles) != 1:
            raise ValueError("Phase 1.0 requires exactly one mobile object.")

        self.scene_config = scene_config

    @staticmethod
    def _position(world) -> np.ndarray:
        mobile = next(iter(world.mobiles.values()))
        position = mobile.get_pos().detach().cpu().numpy()
        if position.ndim == 2:  # right now don't use batches / multiple envs -- so if batch dim present, just return one
            position = position[0]
        return np.asarray(position, dtype=np.float32).copy()

    def collect(self, steps: int) -> np.ndarray:
        if steps < 5:
            raise ValueError("Collect at least 5 simulator steps.")

        from sim.build_world import build_world

        world, _ = build_world(
            self.scene_config,
            show_viewer=False,
            n_envs=1,
            ghost_count=0,
        )
        positions = [self._position(world)]
        for _ in range(steps):
            world.step()
            positions.append(self._position(world))
        return np.stack(positions)


class PositionHistoryDataset(Dataset):
    """Creates a dataset where each sample is (Features=positions across 5 sequential points in time,
     and target=position)."""

    def __init__(self, positions: np.ndarray, history_length: int = 5):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (frames, 3)")
        if len(positions) <= history_length:
            raise ValueError("Need more frames than the history length.")

        self.X = np.stack(
            [positions[i : i + history_length] for i in range(len(positions) - history_length)]
        )
        self.targets = positions[history_length:]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.X[index]),
            torch.from_numpy(self.targets[index]),
        )
    
def create_trajectory_dataset(scene_config, 
                              seed, 
                              steps,
                              n_trajectories,
                              required_history):
    '''
    create n_trajectories randomized rollouts for teaching a model basic rolling physics
    '''

    random_generator = np.random.default_rng(None if seed == 0 else seed)
    trajectories = []
    trajectory_datasets = []
    for _ in range(n_trajectories):
        randomized_scene_config = create_randomized_rolling_scene(
            scene_config,
            random_generator,
        )
        positions = SimulationTrajectoryCollector(randomized_scene_config).collect(steps)
        trajectories.append(positions)
        trajectory_datasets.append(PositionHistoryDataset(positions, required_history))

    return trajectories, ConcatDataset(trajectory_datasets)