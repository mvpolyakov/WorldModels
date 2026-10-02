import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_mlp import (
    _add_random_walk_velocity_noise,
    _evaluate_full_trajectory_rollouts,
    train_model,
)
from training.trajectories import (
    PositionHistoryDataset,
    SimulationTrajectoryCollector,
    create_trajectory_dataset,
)
from world_models import PositionHistoryMLP


class ConstantPositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(3))

    def forward(self, histories, scene_features):
        return self.position.expand(len(histories), -1)


class IncrementingPositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.ones(3))

    def forward(self, histories, scene_features):
        return histories[:, -1] + self.offset


class RecordingPositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(3))
        self.histories = []

    def forward(self, histories, scene_features):
        self.histories.append(histories.detach().clone())
        return self.position.expand(len(histories), -1)


class TrainingTest(unittest.TestCase):
    def test_position_history_model_predict_integrates_displacement(self):
        model = PositionHistoryMLP(
            history_length=2,
            hidden_size=4,
            scene_feature_size=1,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-1].bias.copy_(torch.tensor([0.5, -0.5, 1.0]))

        history = np.asarray(
            [[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]],
            dtype=np.float32,
        )
        predicted_positions = model.predict(
            history,
            np.zeros((1, 1), dtype=np.float32),
        )

        np.testing.assert_array_equal(
            predicted_positions,
            np.asarray([[[2.5, 2.5, 5.0]]], dtype=np.float32),
        )

    def test_trajectory_collection_culls_each_trajectory_when_stationary(self):
        second_trajectory_positions = [
            0.0,
            0.002,
            0.004,
            0.0045,
            0.005,
            0.0055,
            0.006,
            0.0065,
        ]
        positions = [
            torch.tensor(
                [
                    [step * 0.0005, 0.0, 0.0],
                    [second_trajectory_positions[step], 0.0, 0.0],
                ]
            )
            for step in range(8)
        ]
        position_index = 0

        ball = SimpleNamespace(get_pos=lambda: positions[position_index])
        world = SimpleNamespace(
            mobiles={"ball": ball},
            scene_features=np.zeros((2, 10)),
        )

        def step_world():
            nonlocal position_index
            position_index += 1

        world.step = step_world
        world_builder = SimpleNamespace(
            config=SimpleNamespace(mobiles=[object()]),
            build=lambda **_: (world, None),
        )

        with patch("training.trajectories.MIN_MOVEMENT", 0.001):
            trajectories, _ = SimulationTrajectoryCollector(
                world_builder
            ).collect_randomized_batch(
                steps=7,
                n_environments=2,
                random_generator=np.random.default_rng(1),
            )

        self.assertEqual(len(trajectories[0]), 6)
        self.assertEqual(len(trajectories[1]), 8)

    def test_create_trajectory_dataset_splits_trajectories_80_20(self):
        trajectories = [
            np.full((10, 3), trajectory_index, dtype=np.float32)
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

    def test_position_history_dataset_returns_five_future_frames(self):
        positions = np.arange(36, dtype=np.float32).reshape(12, 3)
        dataset = PositionHistoryDataset(
            positions,
            history_length=5,
            prediction_horizon=5,
            scene_features=np.zeros(10, dtype=np.float32),
        )

        self.assertEqual(len(dataset), 3)
        np.testing.assert_array_equal(dataset[0][0], positions[:5])
        np.testing.assert_array_equal(dataset[0][1], positions[5:10])
        np.testing.assert_array_equal(dataset[2][1], positions[7:12])

    def test_train_model_records_training_and_validation_errors(self):
        histories = torch.zeros((3, 1, 3))
        scene_features = torch.zeros((3, 1))
        training_targets = torch.tensor(
            [[[1.0] * 3] * 5, [[3.0] * 3] * 5, [[5.0] * 3] * 5]
        )
        validation_targets = torch.tensor(
            [[[2.0] * 3] * 5, [[4.0] * 3] * 5, [[6.0] * 3] * 5]
        )
        training_loader = DataLoader(
            TensorDataset(histories, training_targets, scene_features),
            batch_size=2,
        )
        validation_loader = DataLoader(
            TensorDataset(histories, validation_targets, scene_features),
            batch_size=2,
        )

        training_errors, validation_errors, _ = train_model(
            ConstantPositionModel(),
            training_loader,
            validation_loader,
            {"lr": 0.0, "epochs": 1, "input_velocity_innovation_std": 0.0},
            torch.device("cpu"),
        )

        self.assertAlmostEqual(training_errors[0], 35.0 / 3.0)
        self.assertAlmostEqual(validation_errors[0], 56.0 / 3.0)

    def test_train_model_feeds_predictions_back_into_rollout(self):
        histories = torch.zeros((1, 2, 3))
        targets = torch.zeros((1, 5, 3))
        scene_features = torch.zeros((1, 1))
        data_loader = DataLoader(
            TensorDataset(histories, targets, scene_features),
            batch_size=1,
        )

        training_errors, validation_errors, _ = train_model(
            IncrementingPositionModel(),
            data_loader,
            data_loader,
            {"lr": 0.0, "epochs": 1, "input_velocity_innovation_std": 0.0},
            torch.device("cpu"),
        )

        self.assertAlmostEqual(training_errors[0], 11.0)
        self.assertAlmostEqual(validation_errors[0], 11.0)

    def test_random_walk_velocity_noise_accumulates_position_drift(self):
        histories = torch.zeros((1, 4, 3))

        with (
            patch("train_mlp.torch.randn_like", return_value=torch.ones((1, 3, 3))),
            patch("train_mlp.velocity_noise_probability", 1.0),
        ):
            perturbed = _add_random_walk_velocity_noise(histories, 1.0)

        expected_positions = torch.tensor(
            [[[0.0] * 3, [1.0] * 3, [3.0] * 3, [6.0] * 3]]
        )
        torch.testing.assert_close(perturbed, expected_positions)

    def test_random_walk_velocity_noise_leaves_unselected_samples_clean(self):
        histories = torch.zeros((2, 2, 3))
        sample_draws = torch.tensor([[[0.0]], [[1.0]]])

        with (
            patch("train_mlp.torch.randn_like", return_value=torch.ones((2, 1, 3))),
            patch("train_mlp.torch.rand", return_value=sample_draws),
            patch("train_mlp.velocity_noise_probability", 0.5),
        ):
            perturbed = _add_random_walk_velocity_noise(
                histories,
                velocity_innovation_std=0.25,
            )

        expected_positions = histories.clone()
        expected_positions[0, 1] = 0.25
        torch.testing.assert_close(perturbed, expected_positions)

    def test_train_model_perturbs_training_histories_only(self):
        histories = torch.zeros((1, 2, 3))
        targets = torch.zeros((1, 1, 3))
        scene_features = torch.zeros((1, 1))
        data_loader = DataLoader(
            TensorDataset(histories, targets, scene_features),
            batch_size=1,
        )
        model = RecordingPositionModel()

        with (
            patch("train_mlp.torch.randn_like", return_value=torch.ones((1, 1, 3))),
            patch("train_mlp.velocity_noise_probability", 1.0),
        ):
            train_model(
                model,
                data_loader,
                data_loader,
                {
                    "lr": 0.0,
                    "epochs": 1,
                    "input_velocity_innovation_std": 0.25,
                },
                torch.device("cpu"),
            )

        expected_training_history = histories.clone()
        expected_training_history[:, 1] = 0.25
        torch.testing.assert_close(model.histories[0], expected_training_history)
        torch.testing.assert_close(model.histories[1], histories)

    def test_full_trajectory_rollout_records_step_errors(self):
        positions = np.zeros((5, 3), dtype=np.float32)
        trajectory_dataset = PositionHistoryDataset(
            positions,
            history_length=2,
            prediction_horizon=1,
            scene_features=np.zeros(1, dtype=np.float32),
        )

        rollouts = _evaluate_full_trajectory_rollouts(
            IncrementingPositionModel(),
            SimpleNamespace(datasets=[trajectory_dataset]),
            torch.device("cpu"),
        )

        expected_position_errors = np.asarray(
            [[1.0] * 3, [2.0] * 3, [3.0] * 3],
            dtype=np.float32,
        )
        expected_velocity_errors = np.ones((3, 3), dtype=np.float32)
        expected_velocity_error_innovations = np.asarray(
            [[1.0] * 3, [0.0] * 3, [0.0] * 3],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            rollouts[0]["position_errors"],
            expected_position_errors,
        )
        np.testing.assert_array_equal(
            rollouts[0]["velocity_errors"],
            expected_velocity_errors,
        )
        np.testing.assert_array_equal(
            rollouts[0]["velocity_error_innovations"],
            expected_velocity_error_innovations,
        )


if __name__ == "__main__":
    unittest.main()
