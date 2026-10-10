import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval import (
    _load_checkpoint_model,
    _resolve_world_builder_name,
    _validate_scene_feature_size,
    get_args as get_eval_args,
)
from sim.scene_features import (
    OPPOSING_RAMPS_SCENE_FEATURE_SIZE,
    SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
)
from sim.utilities.sim_utils import PredictionWorld
from train import (
    _compute_batch_losses,
    _compute_observation_states,
    _compute_trajectory_baseline_losses,
    _roll_out_model,
    _store_validation_rollouts,
    _compute_consistency_error_sum,
    _process_epoch_pass,
    get_args as get_training_args,
)
from training import TrajectorySequenceDataset, TrajectorySliceDataset
from training_profiling_utils import _compute_gradient_norm
from world_models import LatentStateModel, PositionHistoryMLP

SURFACE_ENCODER_FEATURE_SIZE = 8


def _surface_features(batch_size: int) -> np.ndarray:
    """Create one horizontal plane and ball radius for each batch row."""
    return np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2]], dtype=np.float32),
        batch_size,
        axis=0,
    )


class FrameCountingLatentModel(LatentStateModel):
    def __init__(self):
        super().__init__(
            latent_size=1,
            hidden_size=2,
            warm_up_length=2,
        )

    def get_required_history(self):
        return 2

    def initialize_state(self, batch_size, *, device, dtype):
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)

    def encode_predict_update_helper(
        self,
        state,
        position,
        scene_features,
        *,
        advance,
    ):
        del scene_features
        return state + (10.0 if advance else 0.0) + position[:, :1]

    def forward(self, state):
        next_state = state + 10.0
        return next_state.expand(-1, 3), next_state


class LatentStateTest(unittest.TestCase):
    def test_primary_world_command_line_defaults(self):
        with patch("sys.argv", ["train.py"]):
            self.assertEqual(
                get_training_args().world_builder,
                "opposing-ramps",
            )
        with patch("sys.argv", ["eval.py"]):
            self.assertEqual(get_eval_args().world, "opposing-ramps")

    def test_training_disables_layer_norm_by_default_with_opt_in(self):
        with patch("sys.argv", ["train.py"]):
            self.assertFalse(get_training_args().layer_norm)
        with patch("sys.argv", ["train.py", "--layer-norm"]):
            self.assertTrue(get_training_args().layer_norm)

    def test_eval_resolves_explicit_checkpoint_and_legacy_worlds(self):
        self.assertEqual(
            _resolve_world_builder_name(
                "opposing-ramps", {"world_builder": "simple-rolling"}
            ),
            "opposing-ramps",
        )
        self.assertEqual(
            _resolve_world_builder_name(
                None, {"world_builder": "opposing-ramps"}
            ),
            "opposing-ramps",
        )
        self.assertEqual(
            _resolve_world_builder_name(None, {}),
            "simple-rolling",
        )
        self.assertEqual(
            _resolve_world_builder_name(None, None),
            "opposing-ramps",
        )

    def test_eval_allows_variable_surface_counts_for_latent_model(self):
        model = LatentStateModel(latent_size=4)
        _validate_scene_feature_size(
            model,
            np.zeros((1, OPPOSING_RAMPS_SCENE_FEATURE_SIZE), dtype=np.float32),
            "opposing-ramps",
        )

    def test_eval_rejects_incompatible_history_mlp_scene_feature_width(self):
        model = PositionHistoryMLP(scene_feature_size=2)
        with self.assertRaisesRegex(
            ValueError,
            "produces 19 scene features, but this model expects 2",
        ):
            _validate_scene_feature_size(
                model,
                np.zeros(
                    (1, OPPOSING_RAMPS_SCENE_FEATURE_SIZE),
                    dtype=np.float32,
                ),
                "opposing-ramps",
            )

    def test_standard_loader_uses_dataset_wide_padding(self):
        """Even a batch of one retains dataset padding and the original length."""
        short_trajectory = np.ones((2, 3), dtype=np.float32)
        dataset = TrajectorySequenceDataset(
            [short_trajectory, np.full((5, 3), 2.0, dtype=np.float32)],
            np.zeros((2, 2), dtype=np.float32),
        )
        positions, _, lengths, padding_frame_mask = next(iter(DataLoader(dataset)))
        self.assertEqual(positions.shape, (1, 5, 3))
        torch.testing.assert_close(lengths, torch.tensor([2]))
        torch.testing.assert_close(
            padding_frame_mask, torch.tensor([[False, False, True, True, True]])
        )
        torch.testing.assert_close(positions[0, :2], torch.ones((2, 3)))
        torch.testing.assert_close(positions[0, 2:], torch.zeros((3, 3)))
        np.testing.assert_array_equal(dataset.trajectories[0], short_trajectory)

    def test_trajectory_slices_include_frame_zero_and_pad_short_tails(self):
        trajectory = np.arange(21, dtype=np.float32).reshape(7, 3)
        source_dataset = TrajectorySequenceDataset(
            [trajectory],
            np.asarray([[2.0, 3.0]], dtype=np.float32),
        )
        dataset = TrajectorySliceDataset(
            source_dataset,
            slices_per_trajectory=5,
            max_slice_length=6,
            warm_up_length=2,
            sampling_seed=7,
        )

        metadata = dataset.get_slice_metadata()
        self.assertEqual(metadata.shape, (5, 3))
        np.testing.assert_array_equal(metadata[0], np.asarray([0, 0, 6]))
        self.assertEqual(set(metadata[:, 1]), {0, 1, 2, 3, 4})

        full_positions = dataset[0][0]
        self.assertTrue(np.shares_memory(full_positions.numpy(), trajectory))
        batched_positions = next(iter(DataLoader(dataset, batch_size=5)))[0]
        self.assertEqual(batched_positions.shape, (5, 6, 3))

        tail_index = int(np.flatnonzero(metadata[:, 1] == 4)[0])
        positions, features, length, padding_frame_mask = dataset[tail_index]
        self.assertEqual(length.item(), 3)
        torch.testing.assert_close(positions[:3], torch.from_numpy(trajectory[4:]))
        torch.testing.assert_close(positions[3:], torch.zeros((3, 3)))
        torch.testing.assert_close(features, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(
            padding_frame_mask,
            torch.tensor([False, False, False, True, True, True]),
        )

    def test_trajectory_slice_resampling_is_epoch_deterministic(self):
        trajectory = np.arange(60, dtype=np.float32).reshape(20, 3)
        source_dataset = TrajectorySequenceDataset(
            [trajectory],
            np.zeros((1, 2), dtype=np.float32),
        )
        dataset = TrajectorySliceDataset(
            source_dataset,
            slices_per_trajectory=4,
            max_slice_length=8,
            warm_up_length=3,
            sampling_seed=11,
        )
        epoch_zero_metadata = dataset.get_slice_metadata()

        dataset.resample_slices(epoch=1)
        epoch_one_metadata = dataset.get_slice_metadata()
        repeated_dataset = TrajectorySliceDataset(
            source_dataset,
            slices_per_trajectory=4,
            max_slice_length=8,
            warm_up_length=3,
            sampling_seed=11,
        )
        repeated_dataset.resample_slices(epoch=1)

        np.testing.assert_array_equal(
            epoch_one_metadata,
            repeated_dataset.get_slice_metadata(),
        )
        self.assertFalse(np.array_equal(epoch_zero_metadata, epoch_one_metadata))
        np.testing.assert_array_equal(epoch_one_metadata[0], np.asarray([0, 0, 8]))

        epoch_one_metadata[0, 1] = 5
        self.assertEqual(dataset.get_slice_metadata()[0, 1], 0)

    def test_trajectory_slices_require_distinct_valid_later_starts(self):
        source_dataset = TrajectorySequenceDataset(
            [np.zeros((6, 3), dtype=np.float32)],
            np.zeros((1, 2), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "distinct valid slice starts"):
            TrajectorySliceDataset(
                source_dataset,
                slices_per_trajectory=4,
                max_slice_length=6,
                warm_up_length=3,
                sampling_seed=1,
            )

    def test_checkpoint_dispatch_keeps_model_types_distinct(self):
        latent_model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=3,
        )
        loaded_latent_model = _load_checkpoint_model(
            {
                "model_type": "LatentStateModel",
                "latent_size": 4,
                "hidden_size": 8,
                "warm_up_length": 3,
                "layer_norm_enabled": True,
                "state_dict": latent_model.state_dict(),
            },
            torch.device("cpu"),
        )
        baseline_model = PositionHistoryMLP(
            history_length=3,
            hidden_size=8,
            scene_feature_size=2,
        )
        loaded_baseline_model = _load_checkpoint_model(
            {
                "model_type": "PositionHistoryMLP",
                "history_length": 3,
                "hidden_size": 8,
                "scene_feature_size": 2,
                "state_dict": baseline_model.state_dict(),
            },
            torch.device("cpu"),
        )

        self.assertIsInstance(loaded_latent_model, LatentStateModel)
        self.assertIsInstance(loaded_baseline_model, PositionHistoryMLP)

    def test_checkpoint_loads_passthrough_observation_encoder(self):
        """Restore a latent model whose observation encoder has no parameters."""
        model = LatentStateModel(
            latent_size=4,
            layer_norm_enabled=False,
        )
        loaded_model = _load_checkpoint_model(
            {
                "model_type": "LatentStateModel",
                "latent_size": 4,
                "hidden_size": model.hidden_size,
                "scene_feature_size": SURFACE_ENCODER_FEATURE_SIZE,
                "warm_up_length": model.warm_up_length,
                "state_dict": model.state_dict(),
            },
            torch.device("cpu"),
        )
        position = torch.randn((2, 3))
        scene_features = torch.from_numpy(_surface_features(2))
        torch.testing.assert_close(
            loaded_model.encode_observation(position, scene_features),
            model.encode_observation(position, scene_features),
        )
        self.assertFalse(loaded_model.layer_norm_enabled)

    def test_observation_difference_uses_previous_frame_and_resets(self):
        model = LatentStateModel(latent_size=4, hidden_size=8)
        first_positions = torch.zeros((2, 3))
        second_positions = torch.ones((2, 3))
        scene_features = torch.from_numpy(_surface_features(2))

        model.initialize_state(2)
        first_embedding = model.encode_observation(
            first_positions,
            scene_features,
        )
        second_embedding = model.encode_observation(
            second_positions,
            scene_features,
        )
        embedding_size = model.observation_encoder.scene_feature_size

        torch.testing.assert_close(
            first_embedding[:, embedding_size:],
            first_embedding[:, :embedding_size],
        )
        torch.testing.assert_close(
            second_embedding[:, embedding_size:],
            second_embedding[:, :embedding_size]
            - first_embedding[:, :embedding_size],
        )

        model.initialize_state(1)
        reset_embedding = model.encode_observation(
            torch.ones((1, 3)),
            torch.from_numpy(_surface_features(1)),
        )
        torch.testing.assert_close(
            reset_embedding[:, embedding_size:],
            reset_embedding[:, :embedding_size],
        )

    def test_layer_norm_follows_each_hidden_linear_layer(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            layer_norm_enabled=True,
        )

        expected_hidden_layer_counts = {
            model.state_updater: 2,
            model.dynamics_model: 3,
            model.latent_state_decoder: 2,
        }
        for network, hidden_layer_count in expected_hidden_layer_counts.items():
            modules = list(network)
            layer_norm_indices = [
                index
                for index, module in enumerate(modules)
                if isinstance(module, torch.nn.LayerNorm)
            ]
            self.assertEqual(len(layer_norm_indices), hidden_layer_count)
            for index in layer_norm_indices:
                self.assertIsInstance(modules[index - 1], torch.nn.Linear)
                self.assertIsInstance(modules[index + 1], torch.nn.ReLU)
            self.assertIsInstance(modules[-1], torch.nn.Linear)

    def test_mlp_initialization_uses_zero_biases_and_small_output_weights(self):
        model = LatentStateModel(
            latent_size=8,
            hidden_size=4,
            layer_norm_enabled=True,
        )

        for network in (
            model.state_updater,
            model.dynamics_model,
            model.latent_state_decoder,
        ):
            for layer in network:
                if isinstance(layer, torch.nn.Linear):
                    torch.testing.assert_close(
                        layer.bias,
                        torch.zeros_like(layer.bias),
                    )

        output_gains = {
            model.dynamics_model: model.dynamics_output_weight_gain,
            model.latent_state_decoder: model.decoder_output_weight_gain,
        }
        for network, output_gain in output_gains.items():
            output_layer = network[-1]
            fan_out, fan_in = output_layer.weight.shape
            xavier_bound = output_gain * (6.0 / (fan_in + fan_out)) ** 0.5
            self.assertGreater(torch.count_nonzero(output_layer.weight).item(), 0)
            self.assertLessEqual(
                output_layer.weight.abs().max().item(),
                xavier_bound,
            )

    def test_observation_corrects_a_predicted_frame_without_advancing_again(self):
        predicted_world = PredictionWorld(
            np.ones((1, 1, 3), dtype=np.float32),
            FrameCountingLatentModel(),
            _surface_features(1),
        )

        predicted_world.observe(np.full((1, 1, 3), 2.0, dtype=np.float32))
        self.assertEqual(predicted_world.latent_state.item(), 13.0)
        predicted_world.step()
        self.assertEqual(predicted_world.latent_state.item(), 23.0)

        predicted_world.observe(np.full((1, 1, 3), 3.0, dtype=np.float32))

        self.assertEqual(predicted_world.latent_state.item(), 26.0)

    def test_prediction_branches_do_not_mutate_the_starting_state(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
        )
        starting_state = torch.randn((2, 4))
        saved_starting_state = starting_state.clone()

        first_branch = model.predict_state(starting_state)
        second_branch = model.predict_state(starting_state)
        model.predict_state(first_branch)

        torch.testing.assert_close(starting_state, saved_starting_state)
        torch.testing.assert_close(first_branch, second_branch)

    def test_autonomous_rollout_does_not_call_the_observation_encoder(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
        )

        with patch.object(model.observation_encoder, "forward") as encode:
            predicted_position, predicted_state = model(
                model.initialize_state(2)
            )

        encode.assert_not_called()
        self.assertEqual(predicted_position.shape, (2, 3))
        self.assertEqual(predicted_state.shape, (2, 4))

    def test_latent_decoder_returns_displacement(self):
        model = LatentStateModel(
            latent_size=2,
            hidden_size=4,
        )
        with torch.no_grad():
            for parameter in model.latent_state_decoder.parameters():
                parameter.zero_()
            model.latent_state_decoder[-1].bias.copy_(
                torch.tensor([0.5, -0.5, 1.0])
            )

        predicted_displacement = model.decode_state(torch.zeros((1, 2)))

        torch.testing.assert_close(
            predicted_displacement,
            torch.tensor([[0.5, -0.5, 1.0]]),
        )

    def test_latent_rollout_accumulates_decoded_displacements(self):
        model = LatentStateModel(
            latent_size=2,
            hidden_size=4,
            warm_up_length=3,
        )
        with torch.no_grad():
            for parameter in model.latent_state_decoder.parameters():
                parameter.zero_()
            model.latent_state_decoder[-1].bias.copy_(
                torch.tensor([0.5, -0.5, 1.0])
            )

        predicted_positions = _roll_out_model(
            model,
            np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                dtype=np.float32,
            ),
            _surface_features(1),
            prediction_count=3,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            predicted_positions,
            torch.tensor(
                [[[2.5, 1.5, 3.0], [3.0, 1.0, 4.0], [3.5, 0.5, 5.0]]]
            ),
        )

    def test_constant_displacement_decoder_reconstructs_and_forecasts_linear_motion(self):
        model = LatentStateModel(
            latent_size=2,
            hidden_size=4,
            warm_up_length=3,
        )
        with torch.no_grad():
            for parameter in model.latent_state_decoder.parameters():
                parameter.zero_()
            model.latent_state_decoder[-1].bias.fill_(1.0)
        positions = (
            torch.arange(5, dtype=torch.float32)
            .reshape(1, 5, 1)
            .expand(-1, -1, 3)
        )

        losses = _compute_batch_losses(
            model,
            positions,
            torch.from_numpy(_surface_features(1)),
            torch.zeros((1, 5), dtype=torch.bool),
            compute_penetration_loss=False,
        )

        self.assertEqual(losses["reconstruction_loss"].item(), 0.0)
        self.assertEqual(losses["position_loss"].item(), 0.0)

    def test_baselines_use_all_model_forecast_starts_and_horizons(self):
        model = LatentStateModel(
            latent_size=2,
            hidden_size=4,
            warm_up_length=5,
        )
        positions = (
            torch.arange(15, dtype=torch.float32)
            .reshape(1, 15, 1)
            .expand(-1, -1, 3)
        )

        losses = _compute_batch_losses(
            model,
            positions,
            torch.from_numpy(_surface_features(1)),
            torch.zeros((1, 15), dtype=torch.bool),
            compute_penetration_loss=False,
        )

        self.assertEqual(losses["position_element_count"], 165)
        self.assertEqual(losses["constant_velocity_loss"].item(), 0.0)
        self.assertEqual(losses["constant_position_loss"].item(), 22.0)

    def test_trajectory_baseline_losses_change_with_forecast_start(self):
        frame_positions = torch.arange(20, dtype=torch.float32).pow(3)
        positions = frame_positions.reshape(20, 1).expand(-1, 3)

        constant_position_losses, constant_velocity_losses = (
            _compute_trajectory_baseline_losses(
                positions,
                history_length=5,
            )
        )

        self.assertTrue(
            torch.all(torch.diff(constant_position_losses[:6]) > 0)
        )
        self.assertTrue(
            torch.all(torch.diff(constant_velocity_losses[:6]) > 0)
        )
        self.assertTrue(torch.all(torch.isnan(constant_position_losses[6:])))
        self.assertTrue(torch.all(torch.isnan(constant_velocity_losses[6:])))

    def test_validation_rollout_storage_supports_scalar_losses(self):
        trajectory_rollouts = [
            {
                "position_errors": np.zeros((2, 3), dtype=np.float32),
                "constant_position_losses": np.asarray(
                    [1.0, 2.0],
                    dtype=np.float32,
                ),
            },
            {
                "position_errors": np.zeros((1, 3), dtype=np.float32),
                "constant_position_losses": np.asarray([3.0], dtype=np.float32),
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            _store_validation_rollouts(
                trajectory_rollouts,
                Path(temporary_directory),
            )
            saved_path = Path(temporary_directory) / "validation_rollouts.npz"
            with np.load(saved_path) as saved:
                self.assertEqual(saved["position_errors"].shape, (2, 2, 3))
                self.assertEqual(saved["constant_position_losses"].shape, (2, 2))
                self.assertTrue(np.isnan(saved["constant_position_losses"][1, 1]))

    def test_autonomous_rollout_batches_forecast_starts_by_horizon(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=5,
        )
        autonomous_state_shapes = []
        original_predict_state = model.predict_state

        def record_predict_state_shape(state):
            if state.ndim == 3:
                autonomous_state_shapes.append(tuple(state.shape))
            return original_predict_state(state)

        with patch.object(
            model,
            "predict_state",
            side_effect=record_predict_state_shape,
        ):
            _compute_batch_losses(
                model,
                torch.randn((2, 16, 3)),
                torch.from_numpy(_surface_features(2)),
                torch.zeros((2, 16), dtype=torch.bool),
                compute_penetration_loss=False,
            )

        self.assertEqual(
            autonomous_state_shapes,
            [
                (2, active_start_count, 4)
                for active_start_count in range(11, 1, -1)
            ],
        )

    def test_padding_is_excluded_from_updates_and_losses(self):
        dataset = TrajectorySequenceDataset(
            [
                np.arange(18, dtype=np.float32).reshape(6, 3),
                np.arange(24, dtype=np.float32).reshape(8, 3),
            ],
            _surface_features(2),
        )
        positions, features, lengths, padding_frame_mask = next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=2,
                )
            )
        )
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=3,
        )

        losses = _compute_batch_losses(
            model,
            positions,
            features,
            padding_frame_mask,
            compute_penetration_loss=False,
        )
        changed_padding = positions.clone()
        changed_padding[padding_frame_mask] = 1e6
        changed_losses = _compute_batch_losses(
            model,
            changed_padding,
            features,
            padding_frame_mask,
            compute_penetration_loss=False,
        )

        torch.testing.assert_close(lengths, torch.tensor([6, 8]))
        torch.testing.assert_close(
            padding_frame_mask,
            torch.tensor([
                [False, False, False, False, False, False, True, True],
                [False, False, False, False, False, False, False, False],
            ]),
        )
        for name in (
            "position_loss",
            "constant_position_loss",
            "constant_velocity_loss",
            "reconstruction_loss",
            "consistency_loss",
        ):
            torch.testing.assert_close(losses[name], changed_losses[name])

        observation_states = _compute_observation_states(
            model, changed_padding, features, padding_frame_mask
        )
        torch.testing.assert_close(observation_states[0, 6], observation_states[0, 5])
        torch.testing.assert_close(observation_states[0, 7], observation_states[0, 5])
        loss_names = ("position_loss", "reconstruction_loss", "consistency_loss")
        original_gradients = torch.autograd.grad(
            sum(losses[name] for name in loss_names), model.parameters()
        )
        changed_gradients = torch.autograd.grad(
            sum(changed_losses[name] for name in loss_names), model.parameters()
        )
        for original_gradient, changed_gradient in zip(original_gradients, changed_gradients):
            torch.testing.assert_close(original_gradient, changed_gradient)

    def test_single_frame_trajectory_has_no_transition_or_forecast_loss(self):
        """A single observation has no displacement target or forecast."""
        model = LatentStateModel(
            latent_size=4,
        )
        losses = _compute_batch_losses(
            model,
            torch.ones((1, 1, 3)),
            torch.from_numpy(_surface_features(1)),
            torch.zeros((1, 1), dtype=torch.bool),
            compute_penetration_loss=False,
        )
        self.assertEqual(losses["observed_frame_count"], 1)
        self.assertEqual(losses["reconstruction_element_count"], 0)
        self.assertTrue(torch.isfinite(losses["reconstruction_loss"]))
        self.assertEqual(losses["position_element_count"], 0)
        self.assertEqual(losses["consistency_element_count"], 0)
        self.assertEqual(losses["position_loss"].item(), 0.0)
        self.assertEqual(losses["consistency_loss"].item(), 0.0)

    def test_consistency_loss_detaches_only_the_target(self):
        predictions = torch.ones((2, 3), requires_grad=True)
        targets = torch.zeros((2, 3), requires_grad=True)

        loss = _compute_consistency_error_sum(
            predictions,
            targets,
            torch.tensor([False, True]),
        )
        loss.backward()

        self.assertEqual(loss.item(), 3.0)
        torch.testing.assert_close(
            predictions.grad, torch.tensor([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
        )
        self.assertIsNone(targets.grad)

    def test_reconstruction_loss_trains_state_updater(self):
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=2,
        )
        positions = torch.randn((2, 4, 3))
        features = torch.from_numpy(_surface_features(2))
        padding_frame_mask = torch.zeros((2, 4), dtype=torch.bool)

        losses = _compute_batch_losses(
            model,
            positions,
            features,
            padding_frame_mask,
            compute_penetration_loss=False,
        )
        losses["reconstruction_loss"].backward()

        self.assertEqual(list(model.observation_encoder.parameters()), [])
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.state_updater.parameters())
        )

    def test_zero_position_weight_excludes_position_loss_from_total(self):
        """Keep reporting forecast error when reconstruction is the sole objective."""
        dataset = TrajectorySequenceDataset(
            [np.arange(18, dtype=np.float32).reshape(6, 3)],
            _surface_features(1),
        )
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=5,
        )

        batch_metrics = []
        metrics = _process_epoch_pass(
            model,
            DataLoader(dataset, batch_size=1),
            torch.device("cpu"),
            None,
            0.0,
            1.0,
            0.0,
            batch_metrics,
            penetration_weight=0.0,
        )

        self.assertGreater(metrics["position_loss"], 0.0)
        self.assertEqual(metrics["total_loss"], metrics["reconstruction_loss"])
        self.assertEqual(len(batch_metrics), 1)
        self.assertEqual(batch_metrics[0]["total_loss"], metrics["total_loss"])
        self.assertNotIn("gradient_norm", batch_metrics[0])

    def test_training_batch_metrics_include_pre_step_gradient_norms(self):
        dataset = TrajectorySequenceDataset(
            [np.arange(12, dtype=np.float32).reshape(4, 3)],
            _surface_features(1),
        )
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=3,
        )
        batch_metrics = []

        _process_epoch_pass(
            model,
            DataLoader(dataset, batch_size=1),
            torch.device("cpu"),
            torch.optim.Adam(model.parameters(), lr=0.0),
            1.0,
            1.0,
            0.1,
            batch_metrics,
        )

        self.assertEqual(len(batch_metrics), 1)
        for name in (
            "gradient_norm",
            "state_updater_gradient_norm",
            "dynamics_model_gradient_norm",
            "latent_state_decoder_gradient_norm",
        ):
            self.assertGreaterEqual(batch_metrics[0][name], 0.0)

    def test_gradient_norm_combines_parameter_gradients(self):
        first = torch.nn.Parameter(torch.tensor([0.0]))
        second = torch.nn.Parameter(torch.tensor([0.0]))
        unused = torch.nn.Parameter(torch.tensor([0.0]))
        first.grad = torch.tensor([3.0])
        second.grad = torch.tensor([4.0])

        self.assertEqual(_compute_gradient_norm([first, second, unused]), 5.0)

    def test_training_and_prediction_world_rollouts_match(self):
        torch.manual_seed(4)
        model = LatentStateModel(
            latent_size=4,
            hidden_size=8,
            warm_up_length=3,
        )
        observed_positions = np.arange(9, dtype=np.float32).reshape(3, 3) / 10.0
        scene_features = _surface_features(1)
        expected = _roll_out_model(
            model,
            observed_positions,
            scene_features,
            prediction_count=3,
            device=torch.device("cpu"),
        )[0].detach().numpy()

        predicted_world = PredictionWorld(
            observed_positions[:1, None, :].transpose(1, 0, 2),
            model,
            scene_features,
        )
        predicted_world.observe(observed_positions[1][None, None, :])
        predicted_world.observe(observed_positions[2][None, None, :])
        actual = []
        for _ in range(3):
            predicted_world.step()
            actual.append(predicted_world.mobile_positions[0, 0].copy())

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
