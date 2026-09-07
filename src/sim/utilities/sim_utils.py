from pathlib import Path
import numpy as np
from PIL import Image

import genesis as gs


class PredictionWorld:
    """Prediction-only world containing mobile state and a world model.
        For models that require history, keeps track of that as well 

        Specifically, mobile_positions is the 3d tensor of current positions,
        on step() that's updated with another model's prediction.  position_history
        contains the full history of mobile_positions, with the last value along the 
        time dimension==mobile_positions.  step() updates both.
    """

    def __init__(self, mobile_configs, model, n_envs=1):
        self.model = model
        # list of 3d positions (1 per object)
        initial_positions = [
            np.asarray(config.morph_params["pos"], dtype=float)
            for config in mobile_configs
        ]
        # create 3d tensor: Bxkx3 with k=# of mobiles
        self.mobile_positions = np.repeat(
            np.asarray(initial_positions)[None, :, :],
            n_envs,
            axis=0,
        )
        # history of the positions - with the time dimensions added as 3rd axis
        self.position_history = self.mobile_positions[:, :, None, :].copy()

    def step(self):
        """Advance each predicted mobile by one model step.

        ``mobile_positions`` is a (B, K, 3) array containing all environments
        and mobiles, and the model receives that array directly.
        
        position_history adds a T time dimension to it

        """
        required_history_size = self.model.get_required_history()

        desired_history = self.position_history[:, :, -required_history_size:, :]

        predicted_positions = self.model.predict(desired_history)

        self.position_history = np.concatenate(
            [self.position_history, predicted_positions[:, :, None, :]],
            axis=2,
        )

        self.mobile_positions[...] = predicted_positions


class DisplacementErrorTracker:
    """Track summed displacement error across simulation steps."""

    def __init__(self):
        self.values = []
        self._total_step_error = 0.0

    def update(
        self,
        actual_positions: np.ndarray,
        predicted_positions: np.ndarray,
    ) -> float:
        """Record and return cumulative displacement error.

        Both position arrays have shape (B, K, 3): B environments, K
        mobiles, and XYZ coordinates.
        Each mobile contributes one mean error across environments. The
        current step's errors are averaged across mobiles and added to the
        running total without averaging across simulation steps.
        """
        if actual_positions.shape != predicted_positions.shape:
            raise ValueError("Actual and predicted position shapes must match.")

        displacement = actual_positions[..., :2] - predicted_positions[..., :2]
        mobile_errors = np.linalg.norm(displacement, axis=-1).mean(axis=0)

        step_error = float(np.mean(mobile_errors))
        self._total_step_error += step_error
        self.values.append(self._total_step_error)
        return self._total_step_error


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
        prediction_lateral_offset=(0.0, .5, 0.0),
    ):
        self.scene = scene
        self.actual_mobiles = actual_mobiles
        self.camera = camera
        self.actual_mobile_configs = {config.name: config for config in actual_mobile_configs}
        self.predicted_mobile_configs = {config.name: config for config in predicted_mobile_configs}
        self.predicted_mobile_names = [config.name for config in predicted_mobile_configs]
        self.prediction_lateral_offset = np.asarray(prediction_lateral_offset, dtype=float)
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
        """Place one ghost for every actual and predicted mobile.
            Supports batching across envs
        """
        if self._next_ghost >= self.ghost_count:
            return False
        if not hasattr(self, "predicted_mobiles"):
            raise RuntimeError("Predicted mobiles must be set before capture().")

        for name, mobile in self.actual_mobiles.items():
            self.ghosts[name][self._next_ghost].set_pos(mobile.get_pos())

        for index, name in enumerate(self.predicted_mobile_names):
            position = self.predicted_mobiles[:, index, :]
            self.prediction_ghosts[name][self._next_ghost].set_pos(
                position + self.prediction_lateral_offset
            )

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
