from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import torch

from eval import _load_checkpoint_model
from training_profiling_utils import compute_initialization_gains, main
from world_models import LatentStateModel


class InitializationCalibrationTest(unittest.TestCase):
    """Check scale calibration on short motion sequences with trailing padding."""

    def setUp(self):
        """Build two constant-speed trajectories, one ending early."""
        self.positions = torch.zeros((2, 16, 3))
        self.positions[:, :, 0] = torch.arange(16) * 0.01
        self.positions[:, :, 2] = 0.2
        self.scenes = torch.tensor([[0., 0., 0., 0., 0., 1., 0.2]]).repeat(2, 1)
        self.padding = torch.zeros((2, 16), dtype=torch.bool)
        self.padding[1, 9:] = True

    def test_calibration_matches_motion_and_limits_rollout_without_mutation(self):
        """Every sampled seed meets targets and aggregation preserves RNG state."""
        for layer_norm_enabled in (False, True):
            with self.subTest(layer_norm_enabled=layer_norm_enabled):
                original_rng = torch.get_rng_state().clone()
                report = compute_initialization_gains(
                    lambda: LatentStateModel(
                        latent_size=8,
                        hidden_size=8,
                        layer_norm_enabled=layer_norm_enabled,
                    ),
                    self.positions,
                    self.scenes,
                    self.padding,
                    10,
                    seeds=(7, 8, 9),
                )
                self.assertEqual(report["seed_count"], 3)
                self.assertEqual([item["seed"] for item in report["per_seed"]], [7, 8, 9])
                for seed_report in report["per_seed"]:
                    self.assertAlmostEqual(
                        seed_report["observed_displacement_rms"],
                        0.01 / 3 ** 0.5,
                        places=7,
                    )
                    self.assertAlmostEqual(seed_report["decoder_ratio_after"], 1.0, places=5)
                    self.assertLessEqual(seed_report["rollout_drift_ratio_after"], 0.1)
                    self.assertLessEqual(seed_report["residual_ratio_after"], 0.010001)
                self.assertGreater(report["dynamics_output_weight_gain"], 0)
                self.assertGreater(report["decoder_output_weight_gain"], 0)
                torch.testing.assert_close(torch.get_rng_state(), original_rng)

    def test_padding_does_not_change_recommendations(self):
        """Unobserved trailing values cannot affect gains or reported scales."""
        model_factory = lambda: LatentStateModel(latent_size=8, hidden_size=8)
        reference = compute_initialization_gains(
            model_factory, self.positions, self.scenes, self.padding, 10, seeds=(7, 8),
        )
        self.positions[self.padding] = float("nan")
        changed = compute_initialization_gains(
            model_factory, self.positions, self.scenes, self.padding, 10, seeds=(7, 8),
        )
        self.assertEqual(reference, changed)

    def test_static_or_short_samples_do_not_suggest_zero_gains(self):
        """Missing motion must produce an explanation instead of zero initialization."""
        model_factory = lambda: LatentStateModel(latent_size=8, hidden_size=8)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            compute_initialization_gains(
                model_factory, self.positions * 0, self.scenes, self.padding, 10,
            )
        with self.assertRaisesRegex(ValueError, "No observed transitions"):
            compute_initialization_gains(
                model_factory,
                self.positions[:, :5],
                self.scenes,
                self.padding[:, :5],
                10,
            )

    def test_calibration_rejects_missing_or_duplicate_seeds(self):
        """Each requested run must represent a distinct initialization."""
        model_factory = lambda: LatentStateModel(latent_size=8, hidden_size=8)
        with self.assertRaisesRegex(ValueError, "At least one"):
            compute_initialization_gains(
                model_factory, self.positions, self.scenes, self.padding, 10, seeds=(),
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            compute_initialization_gains(
                model_factory,
                self.positions,
                self.scenes,
                self.padding,
                10,
                seeds=(1, 1),
            )

    def test_checkpoint_restores_custom_gains_and_weights(self):
        """Saved initialization choices accompany the actual learned parameters."""
        model = LatentStateModel(
            latent_size=8, hidden_size=8,
            dynamics_output_weight_gain=0.023,
            decoder_output_weight_gain=0.074,
        )
        checkpoint = {
            "model_type": "LatentStateModel",
            "latent_size": model.latent_size,
            "hidden_size": model.hidden_size,
            "warm_up_length": model.warm_up_length,
            "layer_norm_enabled": model.layer_norm_enabled,
            "dynamics_output_weight_gain": model.dynamics_output_weight_gain,
            "decoder_output_weight_gain": model.decoder_output_weight_gain,
            "state_dict": model.state_dict(),
        }
        restored = _load_checkpoint_model(checkpoint, torch.device("cpu"))
        self.assertEqual(restored.dynamics_output_weight_gain, 0.023)
        self.assertEqual(restored.decoder_output_weight_gain, 0.074)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name])

    def test_main_runs_requested_diagnostic_and_prints_json(self):
        """The sole CLI action prints the returned calibration report."""
        expected_report = {
            "dynamics_output_weight_gain": 0.02,
            "decoder_output_weight_gain": 0.3,
        }
        output = io.StringIO()
        with (
            patch(
                "training_profiling_utils._run_compute_init_gains",
                return_value=expected_report,
            ) as run_compute_init_gains,
            redirect_stdout(output),
        ):
            main(["compute_init_gains"])

        run_compute_init_gains.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), expected_report)
