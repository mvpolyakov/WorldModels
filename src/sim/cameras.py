"""Camera setup helpers for scene-relative views."""

import genesis as gs
import numpy as np


_WORLD_UP = np.array((0.0, 0.0, 1.0), dtype=np.float32)


def _build_camera_pose(position, lookat) -> np.ndarray:
    """Build a camera-to-world pose with world Z as the up direction."""
    return gs.utils.geom.pos_lookat_up_to_T(
        np.asarray(position, dtype=np.float32),
        np.asarray(lookat, dtype=np.float32),
        _WORLD_UP,
    )


def _transform_point(point, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to one three-dimensional point."""
    point = np.asarray(point, dtype=np.float32)
    return transform[:3, :3] @ point + transform[:3, 3]


def transform_cameras(world, camera_config, scene_transform: np.ndarray) -> None:
    """Apply a scene transform to the configured viewer and render cameras.

    @param world: World containing the Genesis scene and stationary camera.
    @param camera_config: Baseline viewer and observation camera configuration.
    @param scene_transform: Float array with shape (4, 4) mapping configured
        scene coordinates into randomized scene coordinates.
    """
    # Transform viewer - more involved

    viewer = getattr(world.scene, "viewer", None)
    if viewer is not None:
        pyrender_viewer = getattr(viewer, "_pyrender_viewer", None)
        trackball = getattr(pyrender_viewer, "_trackball", None)
        rotation_center = None
        if trackball is not None:
            rotation_center = _transform_point(
                trackball._n_target,
                scene_transform,
            )

        viewer_pose = _build_camera_pose(
            camera_config.viewer_pos,
            camera_config.viewer_lookat,
        )
        viewer.set_camera_pose(pose=scene_transform @ viewer_pose)

        # Genesis keeps its orbit center outside the camera pose.
        if rotation_center is not None:
            trackball._target = rotation_center
            trackball._n_target = rotation_center.copy()

    # Transform stationary camera - much simpler

    stationary_pose = _build_camera_pose(
        camera_config.obs_pos,
        camera_config.obs_lookat,
    )
    world.stationary_camera.set_pose(transform=scene_transform @ stationary_pose)


def correct_camera_rotation_direction(viewer) -> None:
    """Reverse Genesis trackball rotation on both mouse-drag axes.

    Genesis does not expose these direction settings, so the trackball class's
    rotation handler is wrapped once and shared by all viewer instances.

    @param viewer: Genesis interactive viewer whose trackball class is modified.
    """
    pyrender_viewer = getattr(viewer, "_pyrender_viewer", None)
    trackball = getattr(pyrender_viewer, "_trackball", None)
    if trackball is None:
        return

    trackball_class = type(trackball)
    if getattr(trackball_class, "_rotation_direction_corrected", False):
        return

    original_drag = trackball_class.drag

    def drag_with_corrected_rotation(self, point):
        if self._state == self.STATE_ROTATE:
            point = np.asarray(point, dtype=np.float32).copy()
            point[:2] = 2.0 * self._pdown[:2] - point[:2]
        return original_drag(self, point)

    trackball_class.drag = drag_with_corrected_rotation
    trackball_class._rotation_direction_corrected = True
