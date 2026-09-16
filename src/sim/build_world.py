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
    ) -> tuple[World, Optional[TrajectoryRecorder]]:
        """Build and configure a world for a specific scenario."""


class SimpleRollingWorldBuilder(WorldBuilder):
    """Build the simple ramp-and-ball world used for rolling experiments."""

    config_file_name = "simple_rolling.json"
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
    ) -> tuple:
        """Create randomized rolling values from the scene configuration."""

        # Each randomized property has one value per environment: (n_environments,).
        friction_values = random_generator.uniform(0.1, 1.0, n_environments)
        ramp_angle_degrees = random_generator.uniform(10.0, 30.0, n_environments)
        ramp_angle_radians = np.deg2rad(ramp_angle_degrees)
        ramp_direction_degrees = random_generator.uniform(-180.0, 180.0, n_environments)
        ramp_direction_radians = np.deg2rad(ramp_direction_degrees)
        distance_up_ramp_fractions = random_generator.uniform(0.55, 0.9, n_environments)
        scene_lateral_translations = random_generator.uniform(
            -2,
            2,
            (n_environments, 2),
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
        ball_clearance = 0.01
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply independent randomized values to a batched Genesis world."""

        ramp_positions, ramp_eulers, ball_positions, friction_values = (
            self._create_randomized_rolling_values(
                random_generator,
                n_environments,
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


_world_builder_classes = {
    "simple-rolling": SimpleRollingWorldBuilder,
}


def load_world_builder(builder_name: str = "simple-rolling") -> WorldBuilder:
    """Create the world builder selected by its command-line name."""

    if builder_name not in _world_builder_classes:
        raise ValueError(
            f"World builder '{builder_name}' not found. "
            f"Available: {list(_world_builder_classes)}"
        )
    return _world_builder_classes[builder_name]()
