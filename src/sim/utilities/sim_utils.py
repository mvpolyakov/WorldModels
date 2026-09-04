from pathlib import Path
import numpy as np
from PIL import Image

import genesis as gs


class DummyWorldModel:
    """Generate a small random-walk prediction from an initial position."""

    def __init__(self, seed=0, offset_scale=0.03):
        self._random = np.random.default_rng(seed)
        self.offset_scale = offset_scale

    def predict(self, current_state: list):
        """
        Takes the current world state (for now a list of object positions)
        and returns the updated state 
        """
        new_state = []
        for obj_state in current_state:
            start = np.asarray(obj_state, dtype=float)
            offset = self._random.normal(0.0, self.offset_scale, size=2)
            new_state.append(start + np.array((offset[0], offset[1], 0.0)))

        return new_state
    
class PredictionWorld:
    """Prediction-only world containing mobile state and a world model."""

    def __init__(self, mobile_configs, model, lateral_offset=(0.0, 1.5, 0.0)):
        offsets = np.asarray(lateral_offset, dtype=float)
        self.model = model
        self.mobile_positions = []
        for config in mobile_configs:
            position = np.asarray(config.morph_params["pos"], dtype=float)
            self.mobile_positions.append(position + offsets)

    def step(self):
        """Advance each predicted mobile by one model step."""
        predicted_positions = self.model.predict(self.mobile_positions)
        # so that trajectory capture which keeps track of *the positions* 
        # (not the world object) stays abreast of things
        self.mobile_positions[:] = predicted_positions


def correct_camera_rotation_direction(viewer):
    """Enforces expected behavior on camera rotation motion on both axes."""
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


class TrajectoryRecorder:
    """Capture translucent snapshots for multiple actual and predicted mobiles."""

    # During init, pre-create all the ghost objects at the outset and store under the plane (z=-100) 
    # and when time to capture just move them into the desired positions 
    def _add_hidden_group(self, color, configs):
        ghosts_by_name = {}
        for config in configs:
            radius = config.morph_params.get("radius", 0.2)
            ghosts_by_name[config.name] = []
            for _ in range(self.ghost_count):
                ghosts_by_name[config.name].append(
                    self.scene.add_entity(
                        morph=gs.morphs.Sphere(
                            radius=radius,
                            pos=(0.0, 0.0, -100.0),
                            fixed=True,
                            collision=False,
                        ),
                        material=gs.materials.Rigid(rho=config.rho),
                        surface=gs.surfaces.Default(color=color),
                    )
                )
        return ghosts_by_name

    def __init__(
        self,
        scene,
        camera,
        actual_mobiles,
        actual_mobile_configs,
        predicted_mobile_configs,
        ghost_count,
    ):
        self.scene = scene
        self.actual_mobiles = actual_mobiles
        self.camera = camera
        self.actual_mobile_configs = {config.name: config for config in actual_mobile_configs}
        self.predicted_mobile_configs = {config.name: config for config in predicted_mobile_configs}
        self.predicted_mobile_names = [config.name for config in predicted_mobile_configs]
        self.ghost_count = ghost_count
        self._next_ghost = 0
        self.ghosts = self._add_hidden_group((0.2, 0.6, 1.0, 0.28), 
                                      actual_mobile_configs)
        self.prediction_ghosts = self._add_hidden_group((1.0, 0.35, 0.1, 0.28),
                                                 predicted_mobile_configs)

    def set_up_predictions(self, predicted_mobiles):
        """Attach mobile objects owned by the separate prediction world.
            These will be used to position the pre-generated ghosts for capture
            at run time.
        """
        self.predicted_mobiles = predicted_mobiles

    def capture(self):
        """Place one ghost for every actual and predicted mobile."""
        if self._next_ghost >= self.ghost_count:
            return False
        if not hasattr(self, "predicted_mobiles"):
            raise RuntimeError("Predicted mobiles must be set before capture().")

        for name, mobile in self.actual_mobiles.items():
            self.ghosts[name][self._next_ghost].set_pos(mobile.get_pos())
        for name, position in zip(self.predicted_mobile_names, self.predicted_mobiles):
            self.prediction_ghosts[name][self._next_ghost].set_pos(position)

        self._next_ghost += 1
        return True

    def save_image(self, path):
        """Render the current live-viewer camera pose and save its RGB image."""
        viewer = self.scene.viewer
        self.camera.set_pose(pos=viewer.camera_pos, lookat=viewer.camera_lookat)
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
