import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.modules.setdefault("genesis", SimpleNamespace())

from sim.utilities.sim_utils import TrajectoryRecorder


class TrajectoryRecorderTest(unittest.TestCase):
    def test_prediction_offset_preserves_height_for_rotated_ramps(self):
        starting_positions = np.array([[[0.3, -0.2, 1.4]]])

        for ramp_direction_degrees in (-135.0, -15.5, 0.0, 12, 47.0, 163.0):
            with self.subTest(ramp_direction_degrees=ramp_direction_degrees):
                trajectory_recorder = TrajectoryRecorder.__new__(TrajectoryRecorder)
                trajectory_recorder.prediction_lateral_offset = np.array([0.0, 0.5, 0.0])
                ramp_config = SimpleNamespace(
                    name="ramp",
                    morph_params={"euler": (0.0, -20.0, ramp_direction_degrees)},
                )
                scene_config = SimpleNamespace(landscape=[ramp_config])

                trajectory_recorder.set_up_predictions(starting_positions, scene_config)

                offset_starting_positions = (
                    trajectory_recorder.predicted_mobiles
                    + trajectory_recorder.prediction_lateral_offset
                )
                np.testing.assert_allclose(
                    offset_starting_positions[..., 2],
                    starting_positions[..., 2],
                )


if __name__ == "__main__":
    unittest.main()
