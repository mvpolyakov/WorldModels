from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PositionHistoryMLP(nn.Module):
    """Predict the next XYZ position from a fixed history of positions."""

    def __init__(self, history_length: int = 5, hidden_size: int = 64):
        super().__init__()
        self.history_length = history_length
        self.network = nn.Sequential(
            nn.Linear(history_length * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def get_required_history(self) -> int:
        return self.history_length

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        '''
        history: (Batch x objects - premultiplied) x history x 3(xyz) tensor 
        '''
        if history.ndim != 3 or history.shape[1:] != (self.history_length, 3):
            raise ValueError(
                f"history must have shape (batch, {self.history_length}, 3)"
            )
        # flatten history x xyz into one dimension
        return self.network(history.reshape(history.shape[0], -1))

    # just a helpful method for inference - when dealing with simulator at each step -- not for training
    @torch.inference_mode()
    def predict(self, history: np.ndarray) -> np.ndarray:
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
        positions = self(tensor.reshape(batch_size * n_objects, self.history_length, 3))
        return positions.reshape(batch_size, n_objects, 3).cpu().numpy()
