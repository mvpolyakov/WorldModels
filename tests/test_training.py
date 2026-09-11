import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import train_model
from training.trajectories import create_trajectory_dataset


class ConstantPositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(3))

    def forward(self, histories, scene_features):
        return self.position.expand(len(histories), -1)


class TrainingTest(unittest.TestCase):
    def test_create_trajectory_dataset_splits_trajectories_80_20(self):
        trajectories = [
            np.full((7, 3), trajectory_index, dtype=np.float32)
            for trajectory_index in range(10)
        ]
        scene_features = np.stack(
            [np.full(10, trajectory_index, dtype=np.float32) for trajectory_index in range(10)]
        )

        with patch("training.trajectories.SimulationTrajectoryCollector") as collector_type:
            collector_type.return_value.collect_randomized_batch.return_value = (
                trajectories,
                scene_features,
            )
            collected, training_dataset, validation_dataset = create_trajectory_dataset(
                world_builder=object(),
                seed=1,
                steps=6,
                n_trajectories=10,
                required_history=5,
            )

        self.assertIs(collected, trajectories)
        self.assertEqual(len(training_dataset.datasets), 8)
        self.assertEqual(len(validation_dataset.datasets), 2)
        np.testing.assert_array_equal(
            training_dataset.datasets[-1].scene_features,
            scene_features[7],
        )
        np.testing.assert_array_equal(
            validation_dataset.datasets[0].scene_features,
            scene_features[8],
        )

    def test_train_model_records_training_and_validation_errors(self):
        histories = torch.zeros((3, 1, 3))
        scene_features = torch.zeros((3, 1))
        training_targets = torch.tensor([[1.0] * 3, [3.0] * 3, [5.0] * 3])
        validation_targets = torch.tensor([[2.0] * 3, [4.0] * 3, [6.0] * 3])
        training_loader = DataLoader(
            TensorDataset(histories, training_targets, scene_features),
            batch_size=2,
        )
        validation_loader = DataLoader(
            TensorDataset(histories, validation_targets, scene_features),
            batch_size=2,
        )

        training_errors, validation_errors = train_model(
            ConstantPositionModel(),
            training_loader,
            validation_loader,
            {"lr": 0.0, "epochs": 1},
            torch.device("cpu"),
        )

        self.assertAlmostEqual(training_errors[0], 35.0 / 3.0)
        self.assertAlmostEqual(validation_errors[0], 56.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
