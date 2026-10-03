import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sim.build_world import (
    OpposingRampsWorldBuilder,
    SimpleRollingWorldBuilder,
    load_world_builder,
)
from sim.cameras import transform_cameras
from sim.scene_features import (
    OPPOSING_RAMPS_SCENE_FEATURE_SIZE,
    SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
)
from sim.utilities.viz_utils import draw_ground_coordinate_lines


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


class OpposingRampsRandomizationTest(unittest.TestCase):
    def test_randomized_ramps_have_a_gap_and_ball_starts_on_one_surface(self):
        world_builder = OpposingRampsWorldBuilder()
        (
            ramp_a_positions,
            ramp_a_eulers,
            ramp_b_positions,
            ramp_b_eulers,
            ball_positions,
            gap_center_positions,
            common_yaws,
        ) = world_builder._create_randomized_opposing_ramp_values(
            np.random.default_rng(7),
            n_environments=100,
        )
        ramp_a_tangents, ramp_a_lateral_axes, ramp_a_normals = (
            world_builder._get_surface_axes(
                ramp_a_eulers[:, 1], ramp_a_eulers[:, 2]
            )
        )
        ramp_b_tangents, ramp_b_lateral_axes, ramp_b_normals = (
            world_builder._get_surface_axes(
                ramp_b_eulers[:, 1], ramp_b_eulers[:, 2]
            )
        )
        ramp_length, ramp_width, ramp_thickness = (
            world_builder.ramp_a_config.morph_params["size"]
        )
        ramp_a_inner_edges = (
            ramp_a_positions
            - ramp_length / 2.0 * ramp_a_tangents
            + ramp_thickness / 2.0 * ramp_a_normals
        )
        ramp_b_inner_edges = (
            ramp_b_positions
            - ramp_length / 2.0 * ramp_b_tangents
            + ramp_thickness / 2.0 * ramp_b_normals
        )
        common_yaw_radians = np.deg2rad(common_yaws)
        common_axes = np.column_stack(
            (
                np.cos(common_yaw_radians),
                np.sin(common_yaw_radians),
                np.zeros_like(common_yaw_radians),
            )
        )
        inner_edge_separation_vectors = ramp_b_inner_edges - ramp_a_inner_edges
        inner_edge_center_separations = np.sum(
            inner_edge_separation_vectors * common_axes,
            axis=1,
        )
        np.testing.assert_allclose(
            (ramp_a_inner_edges + ramp_b_inner_edges) / 2.0,
            gap_center_positions,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            inner_edge_separation_vectors,
            np.expand_dims(inner_edge_center_separations, -1) * common_axes,
            atol=1e-6,
        )

        ramp_a_lateral_projections = np.abs(
            np.sum(ramp_a_lateral_axes * common_axes, axis=1)
        )
        ramp_b_lateral_projections = np.abs(
            np.sum(ramp_b_lateral_axes * common_axes, axis=1)
        )
        clear_gap_widths = inner_edge_center_separations - ramp_width / 2.0 * (
            ramp_a_lateral_projections + ramp_b_lateral_projections
        )
        self.assertTrue(
            np.all(
                clear_gap_widths
                >= world_builder.minimum_gap_clearance - 1e-6
            )
        )
        self.assertTrue(
            np.all(
                clear_gap_widths
                <= ramp_width * world_builder.maximum_gap_width_fraction + 1e-6
            )
        )
        self.assertTrue(np.any(clear_gap_widths > 0.1))

        relative_yaws = (
            ramp_a_eulers[:, 2] - 180.0 - ramp_b_eulers[:, 2] + 180.0
        ) % 360.0 - 180.0
        self.assertTrue(
            np.all(
                np.abs(relative_yaws)
                <= world_builder.maximum_relative_yaw_degrees + 1e-6
            )
        )
        self.assertTrue(np.any(np.abs(relative_yaws) > 1.0))
        self.assertTrue(
            np.any(np.abs(ramp_a_eulers[:, 1] - ramp_b_eulers[:, 1]) > 1.0)
        )

        ball_radius = world_builder.ball_config.morph_params["radius"]
        surface_clearance = ball_radius + world_builder.ball_clearance
        minimum_release_distance = max(
            surface_clearance,
            ramp_length * world_builder.minimum_release_distance_fraction,
        )

        def get_release_coordinates(inner_edges, tangents, lateral_axes, normals):
            """Express each ball position relative to one ramp's inner edge."""
            ball_offsets = ball_positions - inner_edges
            return (
                np.sum(ball_offsets * tangents, axis=1),
                np.sum(ball_offsets * lateral_axes, axis=1),
                np.sum(ball_offsets * normals, axis=1),
            )

        ramp_a_distances, ramp_a_lateral, ramp_a_clearances = (
            get_release_coordinates(
                ramp_a_inner_edges,
                ramp_a_tangents,
                ramp_a_lateral_axes,
                ramp_a_normals,
            )
        )
        ramp_b_distances, ramp_b_lateral, ramp_b_clearances = (
            get_release_coordinates(
                ramp_b_inner_edges,
                ramp_b_tangents,
                ramp_b_lateral_axes,
                ramp_b_normals,
            )
        )
        starts_on_ramp_a = (
            np.isclose(ramp_a_lateral, 0.0, atol=1e-6)
            & np.isclose(ramp_a_clearances, surface_clearance, atol=1e-6)
            & (ramp_a_distances >= minimum_release_distance)
            & (ramp_a_distances <= ramp_length / 2.0)
        )
        starts_on_ramp_b = (
            np.isclose(ramp_b_lateral, 0.0, atol=1e-6)
            & np.isclose(ramp_b_clearances, surface_clearance, atol=1e-6)
            & (ramp_b_distances >= minimum_release_distance)
            & (ramp_b_distances <= ramp_length / 2.0)
        )
        np.testing.assert_array_equal(
            starts_on_ramp_a ^ starts_on_ramp_b,
            np.ones(100, dtype=bool),
        )
        self.assertTrue(np.any(starts_on_ramp_a))
        self.assertTrue(np.any(starts_on_ramp_b))

        opposite_rim_heights = np.where(
            starts_on_ramp_a,
            ramp_b_inner_edges[:, 2]
            + ramp_length * ramp_b_tangents[:, 2]
            + surface_clearance * ramp_b_normals[:, 2],
            ramp_a_inner_edges[:, 2]
            + ramp_length * ramp_a_tangents[:, 2]
            + surface_clearance * ramp_a_normals[:, 2],
        )
        self.assertTrue(
            np.all(
                ball_positions[:, 2]
                <= opposite_rim_heights - world_builder.escape_height_margin + 1e-6
            )
        )

    def test_opposing_ramps_builder_is_registered(self):
        world_builder = load_world_builder("opposing-ramps")

        self.assertIsInstance(world_builder, OpposingRampsWorldBuilder)
        self.assertEqual(
            world_builder.scene_feature_size,
            OPPOSING_RAMPS_SCENE_FEATURE_SIZE,
        )
        self.assertEqual(
            [item.name for item in world_builder.config.landscape],
            ["ramp_a", "ramp_b", "ground"],
        )

    def test_simple_rolling_builder_declares_its_scene_feature_size(self):
        world_builder = load_world_builder("simple-rolling")

        self.assertEqual(
            world_builder.scene_feature_size,
            SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
        )


class CoordinateLineTest(unittest.TestCase):
    def test_draws_different_colored_world_axes(self):
        scene = Mock()
        scene_config = SimpleNamespace(
            landscape=[
                SimpleNamespace(
                    name="ground",
                    morph_params={
                        "pos": (0.0, 0.0, -0.05),
                        "size": (8.0, 6.0, 0.1),
                    },
                )
            ]
        )

        draw_ground_coordinate_lines(scene, scene_config)

        self.assertEqual(scene.draw_debug_line.call_count, 2)
        x_axis_call, y_axis_call = scene.draw_debug_line.call_args_list
        self.assertEqual(x_axis_call.args, ((-4.0, 0.0, 0.01), (4.0, 0.0, 0.01)))
        self.assertEqual(y_axis_call.args, ((0.0, -3.0, 0.01), (0.0, 3.0, 0.01)))
        self.assertNotEqual(
            x_axis_call.kwargs["color"],
            y_axis_call.kwargs["color"],
        )


if __name__ == "__main__":
    unittest.main()
