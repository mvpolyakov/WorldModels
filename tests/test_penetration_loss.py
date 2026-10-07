import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import (
    _compute_batch_losses,
    _compute_penetration_error_sum,
    _process_epoch_pass,
    get_args,
)
from training import TrajectorySequenceDataset
from world_models import LatentStateModel


def _horizontal_surface_features(batch_size: int) -> torch.Tensor:
    """Create one z=0 plane and a radius of 0.2 for each batch row."""
    return torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2]]
    ).repeat(batch_size, 1)


class PenetrationLossTest(unittest.TestCase):
    def test_penalizes_only_penetration_beyond_the_actual_target_gap(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
        )
        target_positions = torch.tensor(
            [
                [0.0, 0.0, 0.20],
                [0.0, 0.0, 0.19],
                [0.0, 0.0, 0.30],
                [0.0, 0.0, 0.20],
            ]
        )
        predicted_positions = torch.tensor(
            [
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.17],
                [0.0, 0.0, 0.18],
                [0.0, 0.0, -0.30],
            ],
            requires_grad=True,
        )

        error_sum = _compute_penetration_error_sum(
            model,
            predicted_positions,
            target_positions,
            _horizontal_surface_features(4),
            torch.tensor([False, False, False, True]),
        )
        error_sum.backward()

        self.assertAlmostEqual(error_sum.item(), 0.0033, places=7)
        torch.testing.assert_close(
            predicted_positions.grad,
            torch.tensor(
                [
                    [0.0, 0.0, -0.10],
                    [0.0, 0.0, -0.04],
                    [0.0, 0.0, -0.04],
                    [0.0, 0.0, 0.00],
                ]
            ),
        )

    def test_target_surface_selection_prevents_switching_away_from_penalty(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
        )
        # The target selects the horizontal plane. The prediction is closer to
        # the vertical plane but remains below the target's horizontal surface.
        scene_features = torch.tensor(
            [[
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.2,
            ]]
        )

        error_sum = _compute_penetration_error_sum(
            model,
            torch.tensor([[-0.8, 0.0, 0.1]]),
            torch.tensor([[0.0, 0.0, 0.2]]),
            scene_features,
            torch.tensor([False]),
        )

        self.assertAlmostEqual(error_sum.item(), 0.01, places=7)

    def test_batch_loss_reports_penetration_and_backpropagates_it(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=5,
        )
        with torch.no_grad():
            for parameter in model.latent_state_decoder.parameters():
                parameter.zero_()
            model.latent_state_decoder[-1].bias[2] = -1.0
        observed_positions = torch.zeros((1, 6, 3))
        observed_positions[:, :, 2] = 0.2

        losses = _compute_batch_losses(
            model,
            observed_positions,
            _horizontal_surface_features(1),
            torch.zeros((1, 6), dtype=torch.bool),
            compute_penetration_loss=True,
        )
        losses["penetration_loss"].backward()

        self.assertEqual(losses["penetration_element_count"], 1)
        self.assertGreater(losses["penetration_loss"].item(), 0.0)
        self.assertLess(model.latent_state_decoder[-1].bias.grad[2].item(), 0.0)

    def test_penetration_weight_is_configurable(self):
        with patch(
            "sys.argv",
            ["train.py", "--penetration-weight", "0.25"],
        ):
            self.assertEqual(get_args().penetration_weight, 0.25)

    def test_epoch_total_applies_the_penetration_weight(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=5,
        )
        with torch.no_grad():
            for parameter in model.latent_state_decoder.parameters():
                parameter.zero_()
            model.latent_state_decoder[-1].bias[2] = -1.0
        positions = np.zeros((6, 3), dtype=np.float32)
        positions[:, 2] = 0.2
        dataset = TrajectorySequenceDataset(
            [positions],
            _horizontal_surface_features(1).numpy(),
        )

        metrics = _process_epoch_pass(
            model,
            DataLoader(dataset, batch_size=1),
            torch.device("cpu"),
            None,
            0.0,
            0.0,
            0.0,
            penetration_weight=2.5,
        )

        self.assertEqual(
            metrics["total_loss"],
            2.5 * metrics["penetration_loss"],
        )


if __name__ == "__main__":
    unittest.main()
