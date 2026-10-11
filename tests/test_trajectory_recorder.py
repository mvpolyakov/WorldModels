import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import genesis as gs
import numpy as np
import torch

from eval import (
    INITIAL_WARM_UP_LENGTH,
    _collect_actual_trajectory,
    _collect_surface_episode_metadata,
    _detect_and_handle_surface_transfer,
    _replay_prediction,
)
from sim.build_world import SimpleRollingWorldBuilder
from sim.utilities.sim_utils import (
    DisplacementErrorTracker,
    PredictionWorld,
    TrajectoryRecorder,
)
from world_models import LatentStateModel


class ThreeFrameModel:
    def get_required_history(self):
        return 3

    def predict(self, position_history, scene_features):
        return position_history[:, :, -1, :] + 1.0


class ScheduledDisplacementLatentModel(LatentStateModel):
    """Emit a prescribed displacement sequence for replay-controller tests."""

    def __init__(self, displacements):
        """Store the displacement emitted by each autonomous prediction."""
        super().__init__(latent_size=1, hidden_size=2, warm_up_length=2)
        self.displacements = list(displacements)
        self.next_displacement = 0

    def initialize_state(self, batch_size, *, device, dtype):
        """Return the inert state used by this deterministic test model."""
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)

    def encode_predict_update_helper(
        self,
        state,
        position,
        scene_features,
        *,
        advance,
    ):
        """Leave the inert state unchanged when observations are supplied."""
        del position, scene_features, advance
        return state

    def forward(self, state):
        """Return the next prescribed displacement and unchanged state."""
        displacement = torch.as_tensor(
            self.displacements[self.next_displacement],
            dtype=state.dtype,
            device=state.device,
        ).expand(len(state), -1)
        self.next_displacement += 1
        return displacement, state


def _two_surface_features():
    """Return packed perpendicular X and Z planes for replay tests."""
    return np.asarray(
        [[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.2,
        ]],
        dtype=np.float32,
    )


def _surface_position(surface_index):
    """Return a point nearest the requested test plane."""
    if surface_index == 0:
        return [0.1, 0.0, 2.0]
    return [2.0, 0.0, 0.1]


class PredictionWorldTest(unittest.TestCase):
    def test_observations_warm_up_history_before_prediction(self):
        predicted_world = PredictionWorld(
            np.zeros((1, 1, 3)),
            ThreeFrameModel(),
            np.zeros((1, 10)),
        )

        self.assertTrue(predicted_world.needs_more_warm_up())
        predicted_world.observe(np.ones((1, 1, 3)))
        self.assertTrue(predicted_world.needs_more_warm_up())
        predicted_world.observe(np.full((1, 1, 3), 2.0))
        self.assertFalse(predicted_world.needs_more_warm_up())

        predicted_world.step()

        np.testing.assert_allclose(predicted_world.mobile_positions, 3.0)

    def test_collects_actual_positions_before_prediction_replay(self):
        mobile = SimpleNamespace(position=np.zeros((1, 3), dtype=np.float32))
        mobile.get_pos = lambda: torch.from_numpy(mobile.position.copy())

        def step():
            mobile.position += 1.0

        surface_features = np.zeros((1, 13), dtype=np.float32)
        world = SimpleNamespace(
            mobiles={"ball": mobile},
            scene_features=surface_features,
            surface_features=surface_features,
            step=step,
        )

        recording = _collect_actual_trajectory(
            world,
            step_count=3,
            warm_up_length=2,
        )

        np.testing.assert_allclose(
            recording["actual_positions"][:, 0, 0, 0],
            [0.0, 1.0, 2.0, 3.0],
        )
        np.testing.assert_array_equal(recording["frame_indices"], np.arange(4))
        np.testing.assert_array_equal(recording["scene_features"], surface_features)

    def test_collects_complete_and_ineligible_surface_episode_windows(self):
        surface_sequence = [0, 0, 1, 1, 0, 1, 1, 0]
        actual_positions = np.asarray(
            [_surface_position(index) for index in surface_sequence],
            dtype=np.float32,
        ).reshape(-1, 1, 1, 3)

        metadata = _collect_surface_episode_metadata(
            actual_positions,
            _two_surface_features(),
            warm_up_length=2,
        )

        self.assertEqual(metadata["scene_feature_schema"], "top_planes_v1")
        np.testing.assert_array_equal(
            metadata["closest_surface_indices"],
            surface_sequence,
        )
        transitions = metadata["surface_transitions"]
        self.assertEqual(
            [
                (
                    transition["source_surface"],
                    transition["destination_surface"],
                    transition["transition_frame"],
                    transition["occurrence"],
                )
                for transition in transitions
            ],
            [(0, 1, 2, 1), (1, 0, 4, 2), (0, 1, 5, 3), (1, 0, 7, 4)],
        )
        np.testing.assert_array_equal(
            transitions[0]["observation_window"],
            actual_positions[2:4],
        )
        self.assertIsNone(transitions[1]["observation_window"])
        np.testing.assert_array_equal(
            transitions[2]["observation_window"],
            actual_positions[5:7],
        )
        self.assertIsNone(transitions[3]["observation_window"])

    def test_replay_records_early_transfer_timing_and_reanchors(self):
        surface_sequence = [0, 0, 0, 0, 1, 1, 1]
        actual_positions = np.asarray(
            [_surface_position(index) for index in surface_sequence],
            dtype=np.float32,
        ).reshape(-1, 1, 1, 3)
        recording = {
            "actual_positions": actual_positions,
            "surface_features": _two_surface_features(),
            "warm_up_length": 2,
        }
        recording.update(
            _collect_surface_episode_metadata(
                actual_positions,
                recording["surface_features"],
                warm_up_length=2,
            )
        )
        mobile = Mock()
        world = SimpleNamespace(
            mobiles={"ball": mobile},
            scene=SimpleNamespace(visualizer=Mock()),
        )
        trajectory_recorder = Mock(
            actual_mobiles={"ball": mobile},
            scene=world.scene,
        )
        model = ScheduledDisplacementLatentModel(
            ([2.0, 0.0, -2.0], [0.0, 0.0, 0.0])
        )

        result = _replay_prediction(
            world,
            trajectory_recorder,
            model,
            recording,
            prediction_count=2,
            ghost_interval=1,
        )

        self.assertEqual(result["prediction_count"], 2)
        self.assertEqual(len(result["error_values"]), 2)
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(
            result["transfer_events"],
            [{
                "segment": 1,
                "prediction_step": 1,
                "predicted_transfer_frame": 2,
                "actual_transition_frame": 4,
                "frame_offset": -2,
            }],
        )
        np.testing.assert_array_equal(
            mobile.set_pos.call_args.args[0],
            actual_positions[6, :, 0, :],
        )

    def test_replay_records_a_late_transfer(self):
        surface_sequence = [0, 0, 1, 1, 1]
        actual_positions = np.asarray(
            [_surface_position(index) for index in surface_sequence],
            dtype=np.float32,
        ).reshape(-1, 1, 1, 3)
        recording = {
            "actual_positions": actual_positions,
            "surface_features": _two_surface_features(),
            "warm_up_length": 2,
        }
        recording.update(
            _collect_surface_episode_metadata(
                actual_positions,
                recording["surface_features"],
                warm_up_length=2,
            )
        )
        mobile = Mock()
        world = SimpleNamespace(
            mobiles={"ball": mobile},
            scene=SimpleNamespace(visualizer=Mock()),
        )
        trajectory_recorder = Mock(
            actual_mobiles={"ball": mobile},
            scene=world.scene,
        )
        model = ScheduledDisplacementLatentModel(
            ([0.0, 0.0, 0.0], [2.0, 0.0, -2.0], [0.0, 0.0, 0.0])
        )

        result = _replay_prediction(
            world,
            trajectory_recorder,
            model,
            recording,
            prediction_count=3,
            ghost_interval=1,
        )

        self.assertEqual(result["transfer_events"][0]["frame_offset"], 1)
        self.assertEqual(result["transfer_events"][0]["predicted_transfer_frame"], 3)

    def test_latent_replay_without_a_transfer_reaches_the_requested_limit(self):
        actual_positions = np.asarray(
            [_surface_position(0)] * 6,
            dtype=np.float32,
        ).reshape(-1, 1, 1, 3)
        recording = {
            "actual_positions": actual_positions,
            "surface_features": _two_surface_features(),
            "warm_up_length": 2,
        }
        recording.update(
            _collect_surface_episode_metadata(
                actual_positions,
                recording["surface_features"],
                warm_up_length=2,
            )
        )
        mobile = Mock()
        world = SimpleNamespace(
            mobiles={"ball": mobile},
            scene=SimpleNamespace(visualizer=Mock()),
        )
        trajectory_recorder = Mock(
            actual_mobiles={"ball": mobile},
            scene=world.scene,
        )
        model = ScheduledDisplacementLatentModel(
            ([0.0, 0.0, 0.0],) * 3
        )

        result = _replay_prediction(
            world,
            trajectory_recorder,
            model,
            recording,
            prediction_count=3,
            ghost_interval=1,
        )

        self.assertEqual(result["prediction_count"], 3)
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["transfer_events"], [])
        self.assertIsNone(result["stop_reason"])

    def test_replay_copies_twenty_initial_frames_before_predicting(self):
        actual_positions = np.arange(22, dtype=float).reshape(22, 1, 1, 1)
        actual_positions = np.repeat(actual_positions, 3, axis=3)
        mobile = Mock()
        world = SimpleNamespace(
            mobiles={"ball": mobile},
            scene_features=np.zeros((1, 10)),
            scene=SimpleNamespace(visualizer=Mock()),
        )
        trajectory_recorder = Mock()

        result = _replay_prediction(
            world,
            trajectory_recorder,
            ThreeFrameModel(),
            {
                "actual_positions": actual_positions,
                "warm_up_length": INITIAL_WARM_UP_LENGTH,
            },
            prediction_count=5,
            ghost_interval=10,
        )

        self.assertEqual(result["prediction_count"], 2)
        np.testing.assert_allclose(result["error_values"], 0.0)
        self.assertEqual(mobile.set_pos.call_count, 21)
        self.assertEqual(world.scene.visualizer.update.call_count, 21)
        trajectory_recorder.set_up_predictions.assert_called_once()

    def test_surface_transfer_restarts_from_next_matching_actual_window(self):
        surface_features = np.asarray(
            [[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                0.2,
            ]],
            dtype=np.float32,
        )
        first_window = np.full((2, 1, 1, 3), 1.0)
        second_window = np.full((2, 1, 1, 3), 2.0)
        recording = {
            "surface_features": surface_features,
            "surface_transitions": [
                {
                    "source_surface": 0,
                    "destination_surface": 1,
                    "transition_frame": 4,
                    "observation_window": first_window,
                },
                {
                    "source_surface": 1,
                    "destination_surface": 0,
                    "transition_frame": 9,
                    "observation_window": second_window,
                },
            ],
        }
        predicted_world = Mock()
        predicted_world.mobile_positions = np.asarray([[[2.0, 0.0, 0.1]]])
        trajectory_recorder = Mock()
        trajectory_recorder.actual_mobiles = {"ball": Mock()}
        replay_state = {
            "previous_surface_index": 0,
            "next_transition_index": 0,
        }

        first_event = _detect_and_handle_surface_transfer(
            predicted_world,
            trajectory_recorder,
            recording,
            replay_state,
        )

        self.assertEqual(first_event["actual_frame_anchor"], 5)
        self.assertEqual(first_event["actual_transition_frame"], 4)
        self.assertIsNone(first_event["stop_reason"])
        self.assertEqual(predicted_world.reset_to_observation.call_count, 1)
        np.testing.assert_array_equal(
            predicted_world.reset_to_observation.call_args.args[0],
            first_window[0],
        )
        self.assertEqual(predicted_world.observe.call_count, 1)
        np.testing.assert_array_equal(
            predicted_world.observe.call_args.args[0],
            first_window[1],
        )
        self.assertEqual(trajectory_recorder.capture_observation.call_count, 2)
        self.assertEqual(replay_state["next_transition_index"], 1)

        predicted_world.mobile_positions = np.asarray([[[0.1, 0.0, 2.0]]])
        second_event = _detect_and_handle_surface_transfer(
            predicted_world,
            trajectory_recorder,
            recording,
            replay_state,
        )

        self.assertEqual(second_event["actual_frame_anchor"], 10)
        self.assertEqual(replay_state["next_transition_index"], 2)

    def test_surface_transfer_stops_on_an_unmatched_transition(self):
        recording = {
            "surface_features": np.asarray(
                [[
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.2,
                ]],
                dtype=np.float32,
            ),
            "surface_transitions": [
                {
                    "source_surface": 1,
                    "destination_surface": 0,
                    "transition_frame": 4,
                    "observation_window": np.zeros((2, 1, 1, 3)),
                }
            ],
        }
        predicted_world = Mock()
        predicted_world.mobile_positions = np.asarray([[[2.0, 0.0, 0.1]]])
        replay_state = {
            "previous_surface_index": 0,
            "next_transition_index": 0,
        }

        event = _detect_and_handle_surface_transfer(
            predicted_world,
            Mock(),
            recording,
            replay_state,
        )

        self.assertIn("does not match", event["stop_reason"])
        predicted_world.reset_to_observation.assert_not_called()

    def test_displacement_error_includes_height(self):
        tracker = DisplacementErrorTracker()

        error = tracker.update(
            np.zeros((1, 1, 3)),
            np.asarray([[[0.0, 0.0, 3.0]]]),
        )

        self.assertEqual(error, 3.0)

    def test_surface_transfer_handler_does_nothing_without_a_transfer(self):
        recording = {
            "surface_features": np.asarray(
                [[
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.2,
                ]],
                dtype=np.float32,
            ),
            "surface_transitions": [],
        }
        predicted_world = Mock()
        predicted_world.mobile_positions = np.asarray([[[0.1, 0.0, 2.0]]])
        replay_state = {
            "previous_surface_index": 0,
            "next_transition_index": 0,
        }

        event = _detect_and_handle_surface_transfer(
            predicted_world,
            Mock(),
            recording,
            replay_state,
        )

        self.assertIsNone(event)
        predicted_world.reset_to_observation.assert_not_called()


class TrajectoryRecorderTest(unittest.TestCase):
    def test_render_image_uses_the_stationary_camera(self):
        stationary_camera = Mock()
        stationary_camera.render.return_value = (
            np.zeros((2, 2, 3)),
            None,
            None,
            None,
        )
        trajectory_recorder = TrajectoryRecorder.__new__(TrajectoryRecorder)
        trajectory_recorder.stationary_camera = stationary_camera

        trajectory_recorder.render_image()

        stationary_camera.render.assert_called_once_with(rgb=True)

    def test_save_image_accepts_an_image_rendered_before_viewer_close(self):
        trajectory_recorder = TrajectoryRecorder.__new__(TrajectoryRecorder)
        trajectory_recorder.render_image = Mock()

        with TemporaryDirectory() as temporary_directory:
            output_path = trajectory_recorder.save_image(
                Path(temporary_directory) / "trajectory.png",
                np.zeros((2, 2, 3), dtype=np.uint8),
            )

            self.assertTrue(output_path.exists())
            trajectory_recorder.render_image.assert_not_called()

    def test_prediction_offset_preserves_height_for_rotated_ramps(self):
        starting_positions = np.array([[[0.3, -0.2, 1.4]]])
        world_builder = SimpleRollingWorldBuilder()
        ramp = SimpleNamespace(get_quat=lambda: np.zeros((1, 4)))
        world = SimpleNamespace(landscape={world_builder.ramp_config.name: ramp})

        for ramp_direction_degrees in (-135.0, -15.5, 0.0, 12, 47.0, 163.0):
            with self.subTest(ramp_direction_degrees=ramp_direction_degrees):
                with patch.object(
                    gs.utils.geom,
                    "quat_to_xyz",
                    return_value=np.array(
                        [[0.0, 0.0, np.deg2rad(ramp_direction_degrees)]]
                    ),
                ):
                    trajectory_recorder = TrajectoryRecorder.__new__(TrajectoryRecorder)
                    captured_positions = []
                    trajectory_recorder.predicted_mobile_names = ["ball"]
                    trajectory_recorder.observation_ghosts = {
                        "ball": [SimpleNamespace(set_pos=captured_positions.append)]
                    }
                    trajectory_recorder._next_observation_ghost = 0
                    trajectory_recorder.set_prediction_lateral_offset(
                        world_builder._get_prediction_lateral_offset(world)
                    )
                    trajectory_recorder.set_up_predictions(starting_positions)
                    trajectory_recorder.capture_observation(starting_positions)

                    np.testing.assert_allclose(
                        captured_positions[0][..., 2],
                        starting_positions[:, 0, 2],
                    )


if __name__ == "__main__":
    unittest.main()
