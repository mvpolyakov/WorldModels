from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from sim.scene_config import SceneConfig


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
