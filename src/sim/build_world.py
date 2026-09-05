"""
Build a World from a SceneConfig.

Iterates over config.landscape and config.mobiles to add entities
to the scene generically. Each ObjectConfig specifies a morph_type
string that maps to a gs.morphs class (Box, Sphere, Mesh, etc.).
"""
from typing import Optional

import genesis as gs

from .dataclasses import World
from .scene_config import SceneConfig, ObjectConfig
from .utilities.sim_utils import (
    correct_camera_rotation_direction,
    PredictionWorld,
    TrajectoryRecorder,
)


# Supported morph types mapped to their gs.morphs class
_MORPH_MAP = {
    "Box": gs.morphs.Box,
    "Sphere": gs.morphs.Sphere,
    "Mesh": gs.morphs.Mesh,
    "URDF": gs.morphs.URDF,
    "MJCF": gs.morphs.MJCF,
}


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


def build_world(
    config: SceneConfig,
    show_viewer: bool = True,
    n_envs: int = 1,
    ghost_count: int = 0,
) -> tuple[World, Optional[TrajectoryRecorder]]:
    """Build a World from a SceneConfig.

    Args:
        config: Scene parameters (landscape, mobiles, cameras).
        n_envs: Number of parallel environment copies.
        ghost_count: Number of translucent ball snapshots to pre-create.

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
    for obj in config.landscape:
        _add_object(scene, obj)

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
        mobile_entities[obj.name] = _add_object(scene, obj)

    # Observation camera
    obs_cam = scene.add_camera(
        res=config.camera.obs_res,
        pos=config.camera.obs_pos,
        lookat=config.camera.obs_lookat,
        # Genesis accepts float FOVs, but its untyped integer default causes
        # Pylance to infer this parameter as int.
        fov=config.camera.obs_fov,  # pyright: ignore[reportArgumentType]
    )

    trackable_obj_name = "ball"
    ball = next((obj for obj in config.mobiles if obj.name == trackable_obj_name), None)
    trajectory_recorder = None
    world = World(scene=scene, mobiles=mobile_entities)

    if ghost_count:
        if ball is None:
            raise Exception('config problem: trajectory recorded expected a ' \
                            '"ball" object but it is missing')

        trajectory_recorder = TrajectoryRecorder(
            scene=scene,
            camera=obs_cam,
            actual_mobiles=mobile_entities,
            actual_mobile_configs=config.mobiles,
            predicted_mobile_configs=config.mobiles,
            ghost_count=ghost_count,
        )

    scene.build(n_envs=n_envs, env_spacing=(2, 2))

    if show_viewer:
        correct_camera_rotation_direction(scene.viewer)

    return world, trajectory_recorder
    
