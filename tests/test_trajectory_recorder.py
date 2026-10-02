import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import genesis as gs
import numpy as np

from sim.build_world import SimpleRollingWorldBuilder
from sim.utilities.sim_utils import PredictionWorld, TrajectoryRecorder


class ThreeFrameModel:
    def get_required_history(self):
        return 3

    def predict(self, position_history, scene_features):
        return position_history[:, :, -1, :] + 1.0


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
