from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sim.scene_features import SIMPLE_ROLLING_SCENE_FEATURE_SIZE


class PositionHistoryMLP(nn.Module):
    """Predict the next XYZ position from position history and scene context."""

    def __init__(
        self,
        history_length: int = 5,
        hidden_size: int = 64,
        scene_feature_size: int = SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
    ):
        super().__init__()
        self.history_length = history_length
        self.scene_feature_size = scene_feature_size
        self.network = nn.Sequential(
            nn.Linear(history_length * 3 + scene_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def get_required_history(self) -> int:
        return self.history_length

    def forward(
        self,
        history: torch.Tensor,
        scene_features: torch.Tensor,
    ) -> torch.Tensor:
        '''
        history: (batch, history, 3) tensor of XYZ positions.
        scene_features: (batch, 10) ramp center, two ramp axes, and ball radius.

        The network parameterizes the transition as a displacement, which is added
        to the final history position before returning the next position.
        '''
        if history.ndim != 3 or history.shape[1:] != (self.history_length, 3):
            raise ValueError(
                f"history must have shape (batch, {self.history_length}, 3)"
            )
        
        if scene_features.ndim != 2 or scene_features.shape != (
            history.shape[0],
            self.scene_feature_size,
        ):
            raise ValueError(
                "scene_features must have shape "
                f"(batch, {self.scene_feature_size})"
            )

        # Keep scene context separate until the final input assembly so the
        # model interface can later be fed by an image-based scene encoder.
        inputs = torch.cat(
            [history.reshape(history.shape[0], -1), scene_features],
            dim=1,
        )
        displacement = self.network(inputs)
        return history[:, -1] + displacement

    # just a helpful method for inference - when dealing with simulator 
    # at each step - handles type conversion and input reshaping -- not for training
    @torch.inference_mode()
    def predict(
        self,
        history: np.ndarray,
        scene_features: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(history, dtype=np.float32)

        if values.ndim != 4 or values.shape[2] < 1 or values.shape[3] != 3:
            raise ValueError("history must have shape (batch, objects, frames, 3)")

        if values.shape[2] < self.history_length:
            padding = np.repeat(
                values[:, :, :1, :],
                self.history_length - values.shape[2],
                axis=2,
            )
            values = np.concatenate([padding, values], axis=2)
        elif values.shape[2] > self.history_length:
            values = values[:, :, -self.history_length :, :]

        batch_size, n_objects = values.shape[:2]
        device = next(self.parameters()).device
        tensor = torch.from_numpy(values).to(device)

        features = np.asarray(scene_features, dtype=np.float32)

        if features.ndim != 2 or features.shape != (
            batch_size,
            self.scene_feature_size,
        ):
            raise ValueError(
                "scene_features must have shape "
                f"({batch_size}, {self.scene_feature_size})"
            )
        features = torch.from_numpy(features).to(device)
        features = features[:, None, :].expand(batch_size, n_objects, -1)
        features = features.reshape(batch_size * n_objects, self.scene_feature_size)

        object_histories = tensor.reshape(
            batch_size * n_objects,
            self.history_length,
            3,
        )
        positions = self(
            object_histories,
            features,
        )
        return positions.reshape(batch_size, n_objects, 3).cpu().numpy()
