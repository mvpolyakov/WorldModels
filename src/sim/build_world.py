"""
Scaffold a World from a SceneConfig.

Iterates over config.landscape and config.mobiles to add entities
to the scene generically. Each ObjectConfig specifies a morph_type
string that maps to a gs.morphs class (Box, Sphere, Mesh, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Optional

import genesis as gs
import numpy as np

from .cameras import correct_camera_rotation_direction, transform_cameras
from .dataclasses import World
from .scene_config import SceneConfig, ObjectConfig
from .scene_features import (
    OPPOSING_RAMPS_SCENE_FEATURE_SIZE,
    SIMPLE_ROLLING_SCENE_FEATURE_SIZE,
    pack_surface_features,
)
from .utilities.sim_utils import TrajectoryRecorder

# Supported morph types mapped to their gs.morphs class
_MORPH_MAP = {
    "Box": gs.morphs.Box,
    "Sphere": gs.morphs.Sphere,
    "Mesh": gs.morphs.Mesh,
    "URDF": gs.morphs.URDF,
    "MJCF": gs.morphs.MJCF,
}


class WorldBuilder(ABC):
    """Interface for constructing a configured, ready-to-run World."""

    config: SceneConfig
    scene_feature_size: int

    def get_surface_features(self, world: World) -> np.ndarray:
        """Collect post-randomization landscape box poses for the top-plane geometry helper.

        Call after scene randomization and before collecting observations.

        @return: Packed surface records and radius shaped (B, 6*S + 1), where
            B is environment count and S is landscape box count, including ground.
        """
        box_centers = []
        rotation_matrices = []
        box_thicknesses = []
        for box_config in self.config.landscape:
            box = world.landscape[box_config.name]
            box_centers.append(box.get_pos().detach().cpu().numpy())
            quaternions = box.get_quat().detach().cpu().numpy()
            rotation_matrices.append(np.asarray(gs.utils.geom.quat_to_R(quaternions)))
            box_thicknesses.append(box_config.morph_params["size"][2])
        ball_radii = np.full(
            (box_centers[0].shape[0], 1),
            self.config.mobiles[0].morph_params["radius"],
            dtype=np.float32,
        )
        return pack_surface_features(
            np.stack(box_centers, axis=1),
            np.stack(rotation_matrices, axis=1),
            np.asarray(box_thicknesses, dtype=np.float32),
            ball_radii,
        )

    @staticmethod
    def _get_randomized_scene_count(
        n_environments: int,
        trajectories_per_scene: int,
    ) -> int:
        """Validate grouped randomization and return the distinct scene count."""
        if trajectories_per_scene < 1:
            raise ValueError("trajectories_per_scene must be positive.")
        if n_environments % trajectories_per_scene != 0:
            raise ValueError(
                "n_environments must be divisible by trajectories_per_scene."
            )
        return n_environments // trajectories_per_scene

    @staticmethod
    def _add_object(scene, obj: ObjectConfig):
        """Add a single ObjectConfig to the scene and return the entity."""
        morph_cls = _MORPH_MAP.get(obj.morph_type)
        if morph_cls is None:
            raise ValueError(
                f"Unknown morph_type '{obj.morph_type}'. "
                f"Supported: {list(_MORPH_MAP)}"
            )

        surface_kwargs = {"color": obj.color}
        if obj.texture is not None:
            surface_kwargs = {
                "diffuse_texture": gs.textures.ImageTexture(
                    image_path=obj.texture,
                )
            }

        return scene.add_entity(
            morph=morph_cls(**obj.morph_params),
            material=gs.materials.Rigid(
                friction=obj.friction,
                friction_torsional=obj.friction_torsional,
                friction_rolling=obj.friction_rolling,
                rho=obj.rho,
            ),
            surface=gs.surfaces.Default(**surface_kwargs), # pyright: ignore[reportArgumentType]
        )

    @staticmethod
    def scaffold_world(
        config: SceneConfig,
        show_viewer: bool = True,
        n_environments: int = 1,
        ghost_count: int = 0,
    ) -> tuple[World, Optional[TrajectoryRecorder]]:
        """Build generic Genesis entities from a SceneConfig.

        Args:
            config: Scene parameters (landscape, mobiles, cameras).
            n_environments: Number of parallel environment copies.
            ghost_count: Number of translucent mobile snapshots to pre-create for trajectory capture

        Returns:
            A fully built World instance ready for stepping.
        """
        scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=config.camera.viewer_pos,
                camera_lookat=config.camera.viewer_lookat,
                camera_fov=config.camera.viewer_fov,
            ),
            sim_options=gs.options.SimOptions(
                gravity=config.gravity,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_torsional_friction=True,
                enable_rolling_friction=True,
            ),
        )

        # Landscape objects, including the configured ground component.
        landscape_entities = {}
        for obj in config.landscape:
            landscape_entities[obj.name] = WorldBuilder._add_object(scene, obj)

        # Preserve Genesis's checkerboard floor appearance without adding another
        # collidable surface on top of the configured ground box.
        if any(obj.name == "ground" for obj in config.landscape):
            scene.add_entity(
                morph=gs.morphs.Plane(
                    pos=(0.0, 0.0, 0.001),
                    plane_size=(20.0, 20.0),
                    tile_size=(1.0, 1.0),
                    collision=False,
                )
            )

        # Mobile objects
        mobile_entities = {}
        for obj in config.mobiles:
            mobile_entities[obj.name] = WorldBuilder._add_object(scene, obj)

        # Stationary camera used to render the trajectory image.
        stationary_camera = scene.add_camera(
            res=config.camera.obs_res,
            pos=config.camera.obs_pos,
            lookat=config.camera.obs_lookat,
            # Genesis accepts float FOVs, but its untyped integer default causes
            # Pylance to infer this parameter as int.
            fov=config.camera.obs_fov,  # pyright: ignore[reportArgumentType]
        )

        trajectory_recorder = None
        world = World(
            scene=scene,
            mobiles=mobile_entities,
            landscape=landscape_entities,
            stationary_camera=stationary_camera,
        )

        if ghost_count:
            # pre generate objects for trajectory capture
            trajectory_recorder = TrajectoryRecorder(
                scene=scene,
                stationary_camera=stationary_camera,
                actual_mobiles=mobile_entities,
                actual_mobile_configs=config.mobiles,
                predicted_mobile_configs=config.mobiles,
                ghost_count=ghost_count,
            )

        scene.build(n_envs=n_environments, env_spacing=(2, 2))

        if show_viewer:
            correct_camera_rotation_direction(scene.viewer)

        return world, trajectory_recorder

    @abstractmethod
    def build(
        self,
        show_viewer: bool = True,
        n_environments: int = 1,
        ghost_count: int = 0,
        random_generator: np.random.Generator | None = None,
        trajectories_per_scene: int = 1,
    ) -> tuple[World, Optional[TrajectoryRecorder]]:
        """Build and configure a world for a specific scenario."""


class SimpleRollingWorldBuilder(WorldBuilder):
    """Build the simple ramp-and-ball world used for rolling experiments."""

    config_file_name = "simple_rolling.json"
    scene_feature_size = SIMPLE_ROLLING_SCENE_FEATURE_SIZE
    prediction_lateral_offset_size = 0.5

    def __init__(self):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs",
            self.config_file_name,
        )
        self.config = SceneConfig.from_json(config_path)
        self.ramp_config = next(obj for obj in self.config.landscape if obj.name == "ramp")
        self.ground_config = next(obj for obj in self.config.landscape if obj.name == "ground")
        self.ball_config = next(obj for obj in self.config.mobiles if obj.name == "ball")

    def _create_randomized_rolling_values(
        self,
        random_generator: np.random.Generator,
        n_environments: int,
        trajectories_per_scene: int = 1,
    ) -> tuple:
        """Create randomized rolling values from the scene configuration."""

        scene_count = self._get_randomized_scene_count(
            n_environments,
            trajectories_per_scene,
        )
        friction_values = np.repeat(
            random_generator.uniform(0.1, 1.0, scene_count),
            trajectories_per_scene,
        )
        ramp_angle_degrees = np.repeat(
            random_generator.uniform(10.0, 30.0, scene_count),
            trajectories_per_scene,
        )
        ramp_angle_radians = np.deg2rad(ramp_angle_degrees)
        ramp_direction_degrees = np.repeat(
            random_generator.uniform(-180.0, 180.0, scene_count),
            trajectories_per_scene,
        )
        ramp_direction_radians = np.deg2rad(ramp_direction_degrees)
        if trajectories_per_scene == 4:
            distance_up_ramp_fractions = np.tile(
                np.linspace(0.55, 0.9, trajectories_per_scene),
                scene_count,
            )
        else:
            distance_up_ramp_fractions = random_generator.uniform(
                0.55,
                0.9,
                n_environments,
            )
        scene_lateral_translations = np.repeat(
            random_generator.uniform(-2, 2, (scene_count, 2)),
            trajectories_per_scene,
            axis=0,
        )

        ramp_length, _, ramp_thickness = self.ramp_config.morph_params["size"]
        ramp_center_x, ramp_center_y, _ = self.ramp_config.morph_params["pos"]
        ground_center_z = self.ground_config.morph_params["pos"][2]
        ground_thickness = self.ground_config.morph_params["size"][2]
        ground_surface_z = ground_center_z + ground_thickness / 2.0
        ramp_half_length = ramp_length / 2.0
        ramp_half_thickness = ramp_thickness / 2.0

        ramp_center_z_values = (
            ground_surface_z
            + ramp_half_length * np.sin(ramp_angle_radians)
            + ramp_half_thickness * np.cos(ramp_angle_radians)
        )
        ramp_euler = self.ramp_config.morph_params.get("euler", (0.0, 0.0, 0.0))
        # Position and Euler batches have shape (n_environments, 3).
        ramp_positions = np.column_stack(
            (
                ramp_center_x + scene_lateral_translations[:, 0],
                ramp_center_y + scene_lateral_translations[:, 1],
                ramp_center_z_values,
            )
        ).astype(np.float32)
        ramp_eulers = np.column_stack(
            (
                np.full(n_environments, ramp_euler[0]),
                -ramp_angle_degrees,
                ramp_direction_degrees,
            )
        ).astype(np.float32)

        distances_up_ramp = ramp_length * distance_up_ramp_fractions
        ball_clearance = 1e-4
        ball_radius = self.ball_config.morph_params["radius"]
        distance_above_ramp_center = ramp_half_thickness + ball_radius + ball_clearance
        ball_horizontal_offsets = (
            (ramp_half_length - distances_up_ramp) * np.cos(ramp_angle_radians)
            + distance_above_ramp_center * np.sin(ramp_angle_radians)
        )
        ball_positions = np.column_stack(
            (
                ramp_positions[:, 0]
                + ball_horizontal_offsets * np.cos(ramp_direction_radians),
                ramp_positions[:, 1]
                + ball_horizontal_offsets * np.sin(ramp_direction_radians),
                ramp_center_z_values
                + (distances_up_ramp - ramp_half_length) * np.sin(ramp_angle_radians)
                + distance_above_ramp_center * np.cos(ramp_angle_radians),
            )
        ).astype(np.float32)

        return ramp_positions, ramp_eulers, ball_positions, friction_values

    def _get_scene_features(self, world, config: SceneConfig) -> np.ndarray:
        """Return ramp center, its first two axes, and ball radius per environment."""

        ramp = world.landscape["ramp"]
        ramp_position = ramp.get_pos().detach().cpu().numpy().astype(np.float32)
        ramp_quaternions = ramp.get_quat().detach().cpu().numpy()
        rotation_matrices = np.asarray(gs.utils.geom.quat_to_R(ramp_quaternions))
        ramp_axis_x = rotation_matrices[:, :, 0].astype(np.float32)
        ramp_axis_y = rotation_matrices[:, :, 1].astype(np.float32)
        mobile_config = next(obj for obj in config.mobiles if obj.name == self.ball_config.name)
        radius = np.full(
            (ramp_position.shape[0], 1),
            mobile_config.morph_params["radius"],
            dtype=np.float32,
        )
        
        return np.concatenate(
            [ramp_position, ramp_axis_x, ramp_axis_y, radius],
            axis=1,
        )

    def _apply_randomized_rolling_environments(
        self,
        world,
        random_generator: np.random.Generator,
        n_environments: int,
        trajectories_per_scene: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply grouped scene values and distinct releases to a batched world."""

        ramp_positions, ramp_eulers, ball_positions, friction_values = (
            self._create_randomized_rolling_values(
                random_generator,
                n_environments,
                trajectories_per_scene,
            )
        )

        world.landscape["ramp"].set_pos(ramp_positions)
        world.landscape["ramp"].set_quat(gs.utils.geom.euler_to_quat(ramp_eulers))
        world.mobiles["ball"].set_pos(ball_positions, zero_velocity=True)
        # Keep the configured friction fixed so training and evaluation use the same values.
        # world.landscape["ramp"].set_friction_ratio(
        #     (friction_values / self.ramp_config.friction)[:, None]
        # )
        # world.mobiles["ball"].set_friction_ratio(
        #     (friction_values / self.ball_config.friction)[:, None]
        # )

        return ramp_positions, ramp_eulers

    def _compute_scene_transform(
        self,
        randomized_ramp_position: np.ndarray,
        randomized_ramp_direction_degrees: float,
    ) -> np.ndarray:
        """Return the planar transform from the configured to randomized scene."""

        original_center = np.asarray(
            (*self.ramp_config.morph_params["pos"][:2], 0.0),
            dtype=np.float32,
        )
        randomized_center = np.asarray(
            (*randomized_ramp_position[:2], 0.0),
            dtype=np.float32,
        )
        original_ramp_euler_degrees = self.ramp_config.morph_params.get(
            "euler",
            (0.0, 0.0, 0.0),
        )
        original_ramp_direction_degrees = original_ramp_euler_degrees[2]

        yaw_change_radians = np.deg2rad(
            randomized_ramp_direction_degrees
            - original_ramp_direction_degrees
        )
        cosine = np.cos(yaw_change_radians)
        sine = np.sin(yaw_change_radians)
        rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scene_transform = np.eye(4, dtype=np.float32)
        scene_transform[:3, :3] = rotation
        scene_transform[:3, 3] = randomized_center - rotation @ original_center
        return scene_transform

    def _get_prediction_lateral_offset(self, world) -> np.ndarray:
        """Return the prediction display offset perpendicular to the ramp."""

        ramp_quaternions = world.landscape[self.ramp_config.name].get_quat()
        ramp_eulers = gs.utils.geom.quat_to_xyz(ramp_quaternions)
        ramp_direction_radians = ramp_eulers[0, 2].item()
        return np.array(
            [
                -np.sin(ramp_direction_radians) * self.prediction_lateral_offset_size,
                np.cos(ramp_direction_radians) * self.prediction_lateral_offset_size,
                0.0,
            ]
        )

    def build(
        self,
        show_viewer: bool = True,
        n_environments: int = 1,
        ghost_count: int = 0,
        random_generator: np.random.Generator | None = None,
        trajectories_per_scene: int = 1,
    ) -> tuple[World, Optional[TrajectoryRecorder]]:
        """Scaffold and configure a simple rolling world."""

        world, trajectory_recorder = self.scaffold_world(
            self.config,
            show_viewer=show_viewer,
            n_environments=n_environments,
            ghost_count=ghost_count,
        )
        if random_generator is not None:
            ramp_positions, ramp_eulers = self._apply_randomized_rolling_environments(
                world,
                random_generator,
                n_environments,
                trajectories_per_scene,
            )
            scene_transform = self._compute_scene_transform(
                ramp_positions[0],
                ramp_eulers[0, 2],
            )
            transform_cameras(world, self.config.camera, scene_transform)
        world.scene_features = self._get_scene_features(
            world,
            self.config,
        )
        if trajectory_recorder is not None:
            trajectory_recorder.set_prediction_lateral_offset(
                self._get_prediction_lateral_offset(world)
            )
        return world, trajectory_recorder


class OpposingRampsWorldBuilder(WorldBuilder):
    """Build an asymmetric opposing-ramp scene with randomized releases."""

    config_file_name = "opposing_ramps.json"
    scene_feature_size = OPPOSING_RAMPS_SCENE_FEATURE_SIZE
    prediction_lateral_offset_size = 0.5 # lateral offset for prediction ghosts
    minimum_ramp_angle_degrees = 10.0
    maximum_ramp_angle_degrees = 30.0
    maximum_scene_translation = 2.0
    maximum_gap_width_fraction = 0.5
    maximum_relative_yaw_degrees = 30.0 # rotation in the xy plane
    minimum_gap_clearance = 0.05
    ball_clearance = 1e-4
    escape_height_margin = 0.02
    minimum_release_distance_fraction = 0.25
    sampling_ramp_length = 4.0
    sampling_ramp_width = 5.0

    def __init__(self):
        """Load the opposing ramps, ground, and ball configuration."""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs",
            self.config_file_name,
        )
        self.config = SceneConfig.from_json(config_path)
        self.ramp_a_config = next(
            obj for obj in self.config.landscape if obj.name == "ramp_a"
        )
        self.ramp_b_config = next(
            obj for obj in self.config.landscape if obj.name == "ramp_b"
        )
        self.ground_config = next(
            obj for obj in self.config.landscape if obj.name == "ground"
        )
        self.ball_config = next(
            obj for obj in self.config.mobiles if obj.name == "ball"
        )

        configured_inner_edge_centers = []
        configured_outward_yaws = []
        for ramp_config in (self.ramp_a_config, self.ramp_b_config):
            ramp_position = np.asarray(
                ramp_config.morph_params["pos"], dtype=np.float32
            )
            ramp_euler = ramp_config.morph_params.get(
                "euler", (0.0, 0.0, 0.0)
            )
            ramp_length = self.sampling_ramp_length
            ramp_thickness = ramp_config.morph_params["size"][2]
            ramp_tangent, _, ramp_normal = self._get_surface_axes(
                np.asarray([ramp_euler[1]], dtype=np.float32),
                np.asarray([ramp_euler[2]], dtype=np.float32),
            )
            configured_inner_edge_centers.append(
                ramp_position
                - ramp_length / 2.0 * ramp_tangent[0]
                + ramp_thickness / 2.0 * ramp_normal[0]
            )
            configured_outward_yaws.append(ramp_euler[2])
        self.configured_gap_center = np.mean(
            configured_inner_edge_centers, axis=0
        )
        self.configured_common_yaw_degrees = 0.5 * (
            configured_outward_yaws[0] - 180.0
            + configured_outward_yaws[1]
        )

    @staticmethod
    def _get_surface_axes(
        pitch_degrees: np.ndarray,
        yaw_degrees: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local X, Y, and Z axes for zero-roll ramp rotations."""
        ramp_angles = np.deg2rad(-pitch_degrees)
        ramp_yaws = np.deg2rad(yaw_degrees)
        cosine_angles = np.cos(ramp_angles)
        sine_angles = np.sin(ramp_angles)
        cosine_yaws = np.cos(ramp_yaws)
        sine_yaws = np.sin(ramp_yaws)

        surface_tangents = np.column_stack(
            (
                cosine_yaws * cosine_angles,
                sine_yaws * cosine_angles,
                sine_angles,
            )
        ).astype(np.float32)
        lateral_axes = np.column_stack(
            (-sine_yaws, cosine_yaws, np.zeros_like(cosine_yaws))
        ).astype(np.float32)
        surface_normals = np.column_stack(
            (
                -cosine_yaws * sine_angles,
                -sine_yaws * sine_angles,
                cosine_angles,
            )
        ).astype(np.float32)
        return surface_tangents, lateral_axes, surface_normals

    def _create_randomized_opposing_ramp_values(
        self,
        random_generator: np.random.Generator,
        n_environments: int,
        trajectories_per_scene: int = 1,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Create grouped ramp scenes and valid ball releases for each trajectory."""
        if n_environments < 1:
            raise ValueError("n_environments must be positive.")
        if (
            self.ramp_a_config.morph_params["size"]
            != self.ramp_b_config.morph_params["size"]
        ):
            raise ValueError("Opposing ramps must have matching dimensions.")

        # Generate trajectories_per_scene number of each scene component

        scene_count = self._get_randomized_scene_count(
            n_environments,
            trajectories_per_scene,
        )
        ramp_angles = np.repeat(
            random_generator.uniform(
                self.minimum_ramp_angle_degrees,
                self.maximum_ramp_angle_degrees,
                (scene_count, 2),
            ).astype(np.float32),
            trajectories_per_scene,
            axis=0,
        )
        common_yaws = np.repeat(
            random_generator.uniform(-180.0, 180.0, scene_count).astype(np.float32),
            trajectories_per_scene,
        )
        yaw_offsets = np.repeat(
            random_generator.uniform(
                -self.maximum_relative_yaw_degrees / 2.0,
                self.maximum_relative_yaw_degrees / 2.0,
                (scene_count, 2),
            ).astype(np.float32),
            trajectories_per_scene,
            axis=0,
        )
        scene_translations = np.repeat(
            random_generator.uniform(
                -self.maximum_scene_translation,
                self.maximum_scene_translation,
                (scene_count, 2),
            ).astype(np.float32),
            trajectories_per_scene,
            axis=0,
        )
        gap_center_positions = np.repeat(
            np.expand_dims(self.configured_gap_center, 0),
            n_environments,
            axis=0,
        )
        gap_center_positions[:, :2] += scene_translations

        ramp_width = self.sampling_ramp_width
        clear_gap_widths = np.repeat(
            random_generator.uniform(
                self.minimum_gap_clearance,
                ramp_width * self.maximum_gap_width_fraction,
                scene_count,
            ).astype(np.float32),
            trajectories_per_scene,
        )
        # Yaw makes each inner-edge corner protrude into the shared travel axis.
        # Add both projections so clear_gap_widths remains empty ground.
        inner_edge_center_separations = clear_gap_widths + ramp_width / 2.0 * (
            np.abs(np.sin(np.deg2rad(yaw_offsets[:, 0])))
            + np.abs(np.sin(np.deg2rad(yaw_offsets[:, 1])))
        )
        common_horizontal_axes = np.column_stack(
            (
                np.cos(np.deg2rad(common_yaws)),
                np.sin(np.deg2rad(common_yaws)),
                np.zeros(n_environments, dtype=np.float32),
            )
        ).astype(np.float32)
        ramp_a_inner_edge_centers = (
            gap_center_positions
            - np.expand_dims(inner_edge_center_separations / 2.0, -1)
            * common_horizontal_axes
        )
        ramp_b_inner_edge_centers = (
            gap_center_positions
            + np.expand_dims(inner_edge_center_separations / 2.0, -1)
            * common_horizontal_axes
        )

        ramp_a_eulers = np.column_stack(
            (
                np.zeros(n_environments, dtype=np.float32),
                -ramp_angles[:, 0],
                common_yaws + 180.0 + yaw_offsets[:, 0],
            )
        ).astype(np.float32)
        ramp_b_eulers = np.column_stack(
            (
                np.zeros(n_environments, dtype=np.float32),
                -ramp_angles[:, 1],
                common_yaws + yaw_offsets[:, 1],
            )
        ).astype(np.float32)
        ramp_a_tangents, _, ramp_a_normals = self._get_surface_axes(
            ramp_a_eulers[:, 1], ramp_a_eulers[:, 2]
        )
        ramp_b_tangents, _, ramp_b_normals = self._get_surface_axes(
            ramp_b_eulers[:, 1], ramp_b_eulers[:, 2]
        )

        ramp_length = self.sampling_ramp_length
        ramp_thickness = self.ramp_a_config.morph_params["size"][2]
        ramp_a_positions = (
            ramp_a_inner_edge_centers
            + ramp_length / 2.0 * ramp_a_tangents
            - ramp_thickness / 2.0 * ramp_a_normals
        ).astype(np.float32)
        ramp_b_positions = (
            ramp_b_inner_edge_centers
            + ramp_length / 2.0 * ramp_b_tangents
            - ramp_thickness / 2.0 * ramp_b_normals
        ).astype(np.float32)

        if trajectories_per_scene == 1:
            starts_on_ramp_a = random_generator.random(n_environments) < 0.5
        else:
            release_index_within_scene = (
                np.arange(n_environments) % trajectories_per_scene
            )
            starts_on_ramp_a = (
                release_index_within_scene < trajectories_per_scene / 2
            )
        selected_tangents = np.where(
            np.expand_dims(starts_on_ramp_a, -1),
            ramp_a_tangents,
            ramp_b_tangents,
        )
        selected_normals = np.where(
            np.expand_dims(starts_on_ramp_a, -1),
            ramp_a_normals,
            ramp_b_normals,
        )
        opposite_tangents = np.where(
            np.expand_dims(starts_on_ramp_a, -1),
            ramp_b_tangents,
            ramp_a_tangents,
        )
        opposite_normals = np.where(
            np.expand_dims(starts_on_ramp_a, -1),
            ramp_b_normals,
            ramp_a_normals,
        )
        selected_inner_edge_centers = np.where(
            np.expand_dims(starts_on_ramp_a, -1),
            ramp_a_inner_edge_centers,
            ramp_b_inner_edge_centers,
        )
        opposite_inner_edge_centers = np.where(
            np.expand_dims(starts_on_ramp_a, -1),
            ramp_b_inner_edge_centers,
            ramp_a_inner_edge_centers,
        )

        ball_radius = self.ball_config.morph_params["radius"]
        surface_clearance = ball_radius + self.ball_clearance
        minimum_release_distances = np.full(
            n_environments,
            max(
                surface_clearance,
                ramp_length * self.minimum_release_distance_fraction,
            ),
            dtype=np.float32,
        )
        opposite_rim_center_heights = (
            opposite_inner_edge_centers[:, 2]
            + ramp_length * opposite_tangents[:, 2]
            + surface_clearance * opposite_normals[:, 2]
        )
        release_base_heights = (
            selected_inner_edge_centers[:, 2]
            + surface_clearance * selected_normals[:, 2]
        )
        maximum_release_distances_for_escape = (
            opposite_rim_center_heights
            - self.escape_height_margin
            - release_base_heights
        ) / selected_tangents[:, 2]
        maximum_release_distances = np.minimum(
            ramp_length / 2.0,
            maximum_release_distances_for_escape,
        ).astype(np.float32)
        if np.any(maximum_release_distances <= minimum_release_distances):
            invalid_indices = np.flatnonzero(
                maximum_release_distances <= minimum_release_distances
            )
            raise ValueError(
                "No admissible inner-half ball release for environments "
                f"{invalid_indices.tolist()}."
            )
        if trajectories_per_scene == 4:
            release_fractions = np.tile(
                np.asarray([0.25, 0.75, 0.25, 0.75], dtype=np.float32),
                scene_count,
            )
            release_distances = (
                minimum_release_distances
                + release_fractions
                * (maximum_release_distances - minimum_release_distances)
            )
        else:
            release_distances = random_generator.uniform(
                minimum_release_distances,
                maximum_release_distances,
            ).astype(np.float32)
        ball_positions = (
            selected_inner_edge_centers
            + np.expand_dims(release_distances, -1) * selected_tangents
            + surface_clearance * selected_normals
        ).astype(np.float32)

        return (
            ramp_a_positions,
            ramp_a_eulers,
            ramp_b_positions,
            ramp_b_eulers,
            ball_positions,
            gap_center_positions,
            common_yaws,
        )

    def _apply_randomized_opposing_ramp_environments(
        self,
        world: World,
        random_generator: np.random.Generator,
        n_environments: int,
        trajectories_per_scene,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply randomized ramp poses and ball releases to a batched world."""
        (
            ramp_a_positions,
            ramp_a_eulers,
            ramp_b_positions,
            ramp_b_eulers,
            ball_positions,
            gap_center_positions,
            common_yaws,
        ) = self._create_randomized_opposing_ramp_values(
            random_generator,
            n_environments,
            trajectories_per_scene,
        )
        world.landscape["ramp_a"].set_pos(ramp_a_positions)
        world.landscape["ramp_a"].set_quat(
            gs.utils.geom.euler_to_quat(ramp_a_eulers)
        )
        world.landscape["ramp_b"].set_pos(ramp_b_positions)
        world.landscape["ramp_b"].set_quat(
            gs.utils.geom.euler_to_quat(ramp_b_eulers)
        )
        world.mobiles["ball"].set_pos(ball_positions, zero_velocity=True)
        return gap_center_positions, common_yaws

    def _get_scene_features(self, world: World) -> np.ndarray:
        """Return both ramp poses and the ball radius in stable A/B order."""
        ramp_features = []
        for ramp_name in ("ramp_a", "ramp_b"):
            ramp = world.landscape[ramp_name]
            ramp_positions = ramp.get_pos().detach().cpu().numpy().astype(np.float32)
            ramp_quaternions = ramp.get_quat().detach().cpu().numpy()
            rotation_matrices = np.asarray(
                gs.utils.geom.quat_to_R(ramp_quaternions)
            )
            ramp_features.extend(
                (
                    ramp_positions,
                    rotation_matrices[:, :, 0].astype(np.float32),
                    rotation_matrices[:, :, 1].astype(np.float32),
                )
            )
        radius = np.full(
            (ramp_features[0].shape[0], 1),
            self.ball_config.morph_params["radius"],
            dtype=np.float32,
        )
        return np.concatenate([*ramp_features, radius], axis=1)

    def _get_prediction_lateral_offset(self, world: World) -> np.ndarray:
        """Return the display offset perpendicular to the shared travel axis - 
        used for the prediction ghosts to be next to actual."""
        ramp_quaternion = world.landscape["ramp_b"].get_quat()[0]
        rotation_matrix = np.asarray(
            gs.utils.geom.quat_to_R(ramp_quaternion.detach().cpu().numpy())
        )
        return rotation_matrix[:, 1] * self.prediction_lateral_offset_size

    def _compute_scene_transform(
        self,
        randomized_gap_center: np.ndarray,
        randomized_yaw_degrees: float,
    ) -> np.ndarray:
        """Return the planar transform from the configured to randomized gap."""
        yaw_change_radians = np.deg2rad(
            randomized_yaw_degrees - self.configured_common_yaw_degrees
        )
        cosine = np.cos(yaw_change_radians)
        sine = np.sin(yaw_change_radians)
        rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scene_transform = np.eye(4, dtype=np.float32)
        scene_transform[:3, :3] = rotation
        scene_transform[:3, 3] = (
            randomized_gap_center
            - rotation @ self.configured_gap_center
        )
        scene_transform[2, 3] = 0.0
        return scene_transform

    def build(
        self,
        show_viewer: bool = True,
        n_environments: int = 1,
        ghost_count: int = 0,
        random_generator: np.random.Generator | None = None,
        trajectories_per_scene: int = 1,
    ) -> tuple[World, Optional[TrajectoryRecorder]]:
        """Scaffold and configure the opposing-ramp scene."""
        world, trajectory_recorder = self.scaffold_world(
            self.config,
            show_viewer=show_viewer,
            n_environments=n_environments,
            ghost_count=ghost_count,
        )
        if random_generator is not None:
            gap_center_positions, common_yaws = (
                self._apply_randomized_opposing_ramp_environments(
                    world,
                    random_generator,
                    n_environments,
                    trajectories_per_scene,
                )
            )
            scene_transform = self._compute_scene_transform(
                gap_center_positions[0], common_yaws[0]
            )
            transform_cameras(world, self.config.camera, scene_transform)
        world.scene_features = self._get_scene_features(world)
        if trajectory_recorder is not None:
            trajectory_recorder.set_prediction_lateral_offset(
                self._get_prediction_lateral_offset(world)
            )
        return world, trajectory_recorder


_world_builder_classes = {
    "simple-rolling": SimpleRollingWorldBuilder,
    "opposing-ramps": OpposingRampsWorldBuilder,
}


def load_world_builder(builder_name: str = "simple-rolling") -> WorldBuilder:
    """Create the world builder selected by its command-line name."""

    if builder_name not in _world_builder_classes:
        raise ValueError(
            f"World builder '{builder_name}' not found. "
            f"Available: {list(_world_builder_classes)}"
        )
    return _world_builder_classes[builder_name]()
