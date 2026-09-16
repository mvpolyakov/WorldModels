from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


from sim.build_world import WorldBuilder

#MIN_MOVEMENT = 0.001 # min object displacement to keep collecting trajectory
MIN_MOVEMENT = 0 # actually do want to have samples of near static objects - this pretty much eliminates it

class SimulationTrajectoryCollector:
    """Collect ball-position trajectories from a batched Genesis world.

        WILL NEED TO UPDATE THIS FOR ANY MOBILES CHANGES
    """

    def __init__(self, world_builder: WorldBuilder):
        if len(world_builder.config.mobiles) != 1:
            raise ValueError("Phase 1.0 requires exactly one mobile object.")

        self.world_builder = world_builder

    def collect_randomized_batch(
        self,
        steps: int,
        n_environments: int,
        random_generator: np.random.Generator,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Collect trajectories and their static ramp/radius features."""
        if steps < 5:
            raise ValueError("Collect at least 5 simulator steps.")

        world, _ = self.world_builder.build(
            show_viewer=False,
            n_environments=n_environments,
            ghost_count=0,
            random_generator=random_generator,
        )

        positions_by_step = [
            world.mobiles["ball"].get_pos().detach().cpu().numpy().copy()
        ]

        stationary_frame_counts = np.zeros(n_environments, dtype=np.int32)
        trajectory_end_indices = np.full(n_environments, -1, dtype=np.int32)

        for _ in range(steps):
            world.step()

            # All of this just keeps track of when each individual trajectory seizes substantial movement for an early cutoff 
            # (don't want a bunch of samples of essentially stationary objects)

            current_positions = (
                world.mobiles["ball"].get_pos().detach().cpu().numpy().copy()
            )
            displacements = np.linalg.norm(
                current_positions - positions_by_step[-1],
                axis=1,
            )
            stationary_frame_counts = np.where(
                displacements < MIN_MOVEMENT,
                stationary_frame_counts + 1,
                0,
            )
            positions_by_step.append(current_positions)
            newly_stationary = (
                (stationary_frame_counts >= 5) & (trajectory_end_indices < 0)
            )
            trajectory_end_indices[newly_stationary] = len(positions_by_step)

        positions = np.stack(positions_by_step)

        # have to collect all trajectories fully but then cut off individual trajectories where movement is too small 
        # they will be seamlessly combined regardless of length in the dataset collection later
        trajectory_end_indices[trajectory_end_indices < 0] = len(positions_by_step)
        trajectories = [
            positions[:trajectory_end_indices[environment_index], environment_index, :]
            for environment_index in range(n_environments)
        ]
        return trajectories, world.scene_features.astype(np.float32)


class PositionHistoryDataset(Dataset):
    """Create position histories and their subsequent rollout targets."""

    def __init__(
        self,
        positions: np.ndarray,
        history_length: int = 5,
        prediction_horizon: int = 5,
        *,
        scene_features: np.ndarray,
    ):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (frames, 3)")
        if len(positions) < history_length + prediction_horizon:
            raise ValueError("Need enough frames for the history and prediction horizon.")
        scene_features = np.asarray(scene_features, dtype=np.float32)
        if scene_features.ndim != 1:
            raise ValueError("scene_features must have shape (features,)")
        self.scene_features = scene_features

        sample_count = len(positions) - history_length - prediction_horizon + 1
        self.X = np.stack(
            [positions[i : i + history_length] for i in range(sample_count)]
        )
        self.targets = np.stack(
            [
                positions[i + history_length : i + history_length + prediction_horizon]
                for i in range(sample_count)
            ]
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = (
            torch.from_numpy(self.X[index]),
            torch.from_numpy(self.targets[index]),
        )
        return sample + (torch.from_numpy(self.scene_features),)
    
def create_trajectory_dataset(world_builder,
                              seed, 
                              steps,
                              n_trajectories,
                              required_history,
                              prediction_horizon=5):
    '''
    create n_trajectories randomized rollouts for teaching a model basic rolling physics
    '''

    if n_trajectories < 2:
        raise ValueError("At least 2 trajectories are required for training and validation.")

    random_generator = np.random.default_rng(None if seed == 0 else seed)
    trajectories, scene_features = SimulationTrajectoryCollector(
        world_builder
    ).collect_randomized_batch(
        steps,
        # 1 trajectory per environment
        n_environments=n_trajectories,
        random_generator=random_generator,
    )
    validation_trajectory_count = max(1, n_trajectories // 5)
    training_trajectory_count = n_trajectories - validation_trajectory_count

    training_trajectory_datasets = [
        PositionHistoryDataset(
            positions,
            required_history,
            prediction_horizon,
            scene_features=features,
        )
        for positions, features in zip(
            trajectories[:training_trajectory_count],
            scene_features[:training_trajectory_count],
        )
    ]
    validation_trajectory_datasets = [
        PositionHistoryDataset(
            positions,
            required_history,
            prediction_horizon,
            scene_features=features,
        )
        for positions, features in zip(
            trajectories[training_trajectory_count:],
            scene_features[training_trajectory_count:],
        )
    ]

    return (
        trajectories,
        ConcatDataset(training_trajectory_datasets),
        ConcatDataset(validation_trajectory_datasets),
    )
