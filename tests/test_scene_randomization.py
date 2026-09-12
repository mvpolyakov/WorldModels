import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sim.build_world import SimpleRollingWorldBuilder
from sim.cameras import transform_cameras


class SceneRandomizationTest(unittest.TestCase):
    def test_randomized_scene_translates_ramp_and_ball_together(self):
        world_builder = SimpleRollingWorldBuilder()
        ramp_positions, ramp_eulers, ball_positions, _ = (
            world_builder._create_randomized_rolling_values(
                np.random.default_rng(4),
                n_environments=20,
            )
        )

        configured_ramp_position = np.asarray(
            world_builder.ramp_config.morph_params["pos"]
        )
        lateral_translations = (
            ramp_positions[:, :2] - configured_ramp_position[:2]
        )
        self.assertTrue(np.any(np.abs(lateral_translations) > 0.1))
        self.assertTrue(
            np.all(
                np.abs(lateral_translations)
                <= 2
            )
        )

        ramp_directions = np.deg2rad(ramp_eulers[:, 2])
        ball_offsets = ball_positions[:, :2] - ramp_positions[:, :2]
        lateral_errors = (
            ball_offsets[:, 0] * np.sin(ramp_directions)
            - ball_offsets[:, 1] * np.cos(ramp_directions)
        )
        np.testing.assert_allclose(lateral_errors, 0.0, atol=1e-6)

    def test_scene_transform_rotates_about_original_ramp_center(self):
        world_builder = SimpleRollingWorldBuilder()
        configured_ramp_position = np.asarray(
            world_builder.ramp_config.morph_params["pos"],
            dtype=np.float32,
        )
        configured_ramp_direction = 30.0
        world_builder.ramp_config.morph_params["euler"][2] = (
            configured_ramp_direction
        )
        randomized_ramp_position = configured_ramp_position + (3.0, -2.0, 0.0)

        scene_transform = world_builder._compute_scene_transform(
            randomized_ramp_position,
            configured_ramp_direction + 90.0,
        )
        configured_points = np.array(
            [
                configured_ramp_position + (2.0, 0.0, 2.0),
                configured_ramp_position + (0.0, 1.0, 0.5),
            ]
        )
        transformed_points = (
            configured_points @ scene_transform[:3, :3].T
            + scene_transform[:3, 3]
        )

        np.testing.assert_allclose(
            transformed_points[0],
            randomized_ramp_position + (0.0, 2.0, 2.0),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            transformed_points[1],
            randomized_ramp_position + (-1.0, 0.0, 0.5),
            atol=1e-6,
        )

    def test_scene_transform_handles_identity_and_xy_translation(self):
        world_builder = SimpleRollingWorldBuilder()
        configured_ramp_position = np.asarray(
            world_builder.ramp_config.morph_params["pos"],
            dtype=np.float32,
        )
        configured_ramp_direction = world_builder.ramp_config.morph_params[
            "euler"
        ][2]

        identity_transform = world_builder._compute_scene_transform(
            configured_ramp_position + (0.0, 0.0, 8.0),
            configured_ramp_direction,
        )
        translation_transform = world_builder._compute_scene_transform(
            configured_ramp_position + (1.0, -2.0, 8.0),
            configured_ramp_direction,
        )

        np.testing.assert_allclose(identity_transform, np.eye(4), atol=1e-6)
        np.testing.assert_allclose(
            translation_transform[:3, :3],
            np.eye(3),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            translation_transform[:3, 3],
            (1.0, -2.0, 0.0),
        )

    def test_cameras_use_their_own_poses_and_transform_orbit_center(self):
        viewer = Mock()
        trackball = SimpleNamespace(
            _target=np.array((0.0, 1.0, 0.0)),
            _n_target=np.array((0.0, 1.0, 0.0)),
        )
        viewer._pyrender_viewer = SimpleNamespace(_trackball=trackball)
        stationary_camera = Mock()
        world = SimpleNamespace(
            scene=SimpleNamespace(viewer=viewer),
            stationary_camera=stationary_camera,
        )
        camera_config = SimpleNamespace(
            viewer_pos=(2.0, 0.0, 2.0),
            viewer_lookat=(0.0, 0.0, 0.0),
            obs_pos=(0.0, -3.0, 1.0),
            obs_lookat=(0.0, 0.0, 0.0),
        )
        scene_transform = np.array(
            [
                [0.0, -1.0, 0.0, 3.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        transform_cameras(world, camera_config, scene_transform)

        viewer_pose = viewer.set_camera_pose.call_args.kwargs["pose"]
        stationary_pose = stationary_camera.set_pose.call_args.kwargs[
            "transform"
        ]
        np.testing.assert_allclose(viewer_pose[:3, 3], (3.0, 0.0, 2.0))
        np.testing.assert_allclose(stationary_pose[:3, 3], (6.0, -2.0, 1.0))
        np.testing.assert_allclose(
            viewer_pose[:3, 2],
            (0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
        )
        np.testing.assert_allclose(trackball._target, (2.0, -2.0, 0.0))
        np.testing.assert_array_equal(trackball._n_target, trackball._target)
        self.assertFalse(
            np.array_equal(
                trackball._target,
                viewer_pose[:3, 3] - viewer_pose[:3, 2],
            )
        )
        self.assertFalse(np.array_equal(viewer_pose, stationary_pose))

    def test_transform_cameras_handles_a_disabled_viewer(self):
        stationary_camera = Mock()
        world = SimpleNamespace(
            scene=SimpleNamespace(viewer=None),
            stationary_camera=stationary_camera,
        )
        camera_config = SimpleNamespace(
            obs_pos=(0.0, -3.0, 1.0),
            obs_lookat=(0.0, 0.0, 0.0),
        )

        transform_cameras(world, camera_config, np.eye(4, dtype=np.float32))

        stationary_camera.set_pose.assert_called_once()

if __name__ == "__main__":
    unittest.main()
