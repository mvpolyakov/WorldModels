from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


from sim.build_world import WorldBuilder


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
    ) -> list[np.ndarray]:
        """Collect independently randomized trajectories in one batched world.

        Return one array of shape (steps + 1, 3) per environment. Each array
        includes the ball's initial position followed by its stepped positions.
        """
        if steps < 5:
            raise ValueError("Collect at least 5 simulator steps.")

        world, _ = self.world_builder.build(
            show_viewer=False,
            n_environments=n_environments,
            ghost_count=0,
            random_generator=random_generator,
        )

        # Each capture has shape (n_environments, 3).
        positions_by_step = [
            world.mobiles["ball"].get_pos().detach().cpu().numpy().copy()
        ]
        for _ in range(steps):
            world.step()
            positions_by_step.append(
                world.mobiles["ball"].get_pos().detach().cpu().numpy().copy()
            )

        # Stacking adds time first: (steps + 1, n_environments, 3).
        positions = np.stack(positions_by_step)
        return [positions[:, environment_index, :] for environment_index in range(n_environments)]


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
    
def create_trajectory_dataset(world_builder,
                              seed, 
                              steps,
                              n_trajectories,
                              required_history):
    '''
    create n_trajectories randomized rollouts for teaching a model basic rolling physics
    '''

    random_generator = np.random.default_rng(None if seed == 0 else seed)
    trajectories = SimulationTrajectoryCollector(world_builder).collect_randomized_batch(
        steps,
        # 1 trajectory per environment
        n_environments=n_trajectories,
        random_generator=random_generator,
    )
    trajectory_datasets = [
        PositionHistoryDataset(positions, required_history)
        for positions in trajectories
    ]

    return trajectories, ConcatDataset(trajectory_datasets)
