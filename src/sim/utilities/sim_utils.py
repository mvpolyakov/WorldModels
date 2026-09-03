from pathlib import Path

from PIL import Image

import genesis as gs


class TrajectoryRecorder:
    """Capture translucent snapshots of a mobile object's trajectory."""

    def __init__(self, scene, camera, mobile, mobile_config, ghost_count):
        self.scene = scene
        self.mobile = mobile
        self.camera = camera
        self.ghosts = []
        self._next_ghost = 0

        radius = mobile_config.morph_params.get("radius", 0.2)
        for _ in range(ghost_count):
            self.ghosts.append(
                scene.add_entity(
                    morph=gs.morphs.Sphere(
                        radius=radius,
                        pos=(0.0, 0.0, -100.0),
                        fixed=True,
                        collision=False,
                    ),
                    material=gs.materials.Rigid(rho=mobile_config.rho),
                    surface=gs.surfaces.Default(
                        color=(0.2, 0.6, 1.0, 0.28),
                    ),
                )
            )

    def capture(self):
        """Place the next ghost at the tracked object's current position."""
        if self._next_ghost >= len(self.ghosts):
            return False

        self.ghosts[self._next_ghost].set_pos(self.mobile.get_pos())
        self._next_ghost += 1
        return True

    def save_image(self, path):
        """Render the current live-viewer camera pose and save its RGB image."""
        viewer = self.scene.viewer
        self.camera.set_pose(
            pos=viewer.camera_pos,
            lookat=viewer.camera_lookat,
        )

        rendered = self.camera.render(rgb=True)
        rgb = rendered[0] if isinstance(rendered, tuple) else rendered
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb.astype("uint8")).save(output_path)
        return output_path
