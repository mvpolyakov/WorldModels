from pathlib import Path
import numpy as np
from PIL import Image
import torch

import genesis as gs

from world_models import LatentStateModel

# Used in evaluation only
class PredictionWorld:
    """Prediction-only world containing mobile state and a world model.
        For models that require history, keeps track of that as well 

        Specifically, mobile_positions is the 3d tensor of current positions,
        on step() that's updated with another model's prediction.  position_history
        contains the full history of mobile_positions, with the last value along the 
        time dimension==mobile_positions. Observations seed the history during warm-up,
        then step() updates both with predictions.

        For reccurent models, accumulate and carry state here
    """

    def __init__(self, initial_mobile_positions, model, scene_features):
        self.model = model
        # Position tensor: BxKx3 with K mobiles.
        self.mobile_positions = np.asarray(initial_mobile_positions, dtype=float).copy()
        # history of the positions - with the time dimensions added as 3rd axis
        self.position_history = self.mobile_positions[:, :, None, :].copy()
        self.scene_features = scene_features
        self._uses_recurrent_state = isinstance(model, LatentStateModel)
        
        if self._uses_recurrent_state:
            # Generate the first latent state by doing the first encoding step (but no prediction)

            self._observed_frame_count = 1

            # Indicates the stored latent represents a prediction for the current frame that hasn’t yet incorporated that frame’s observation.
            # After step(): True—the model has advanced to the next frame by prediction.
            # After observe(): False—the state incorporates the observation.
            self._state_needs_observation_update = False
            batch_size, mobile_count = self.mobile_positions.shape[:2]
            device = next(model.parameters()).device

            features = torch.as_tensor(
                scene_features,
                dtype=torch.float32,
                device=device,
            )
            
            self._raw_scene_features = features[:, None, :].expand(
                batch_size,
                mobile_count,
                -1,
            ).reshape(batch_size * mobile_count, -1)
            
            positions = torch.as_tensor(
                self.mobile_positions,
                dtype=torch.float32,
                device=device,
            ).reshape(batch_size * mobile_count, 3)

            with torch.inference_mode():
                initial_state = model.initialize_state(
                    batch_size * mobile_count,
                    device=device,
                    dtype=positions.dtype,
                )
                self.latent_state = model.encode_predict_update_helper(
                    initial_state,
                    positions,
                    self._raw_scene_features,
                    advance=False,
                )

    def needs_more_warm_up(self):
        """Return whether the model still needs observed history."""
        if self._uses_recurrent_state:
            return self._observed_frame_count < self.model.get_required_history()
        else:
            return self.position_history.shape[2] < self.model.get_required_history()

    def observe(self, observed_mobile_positions):
        """Add observed positions to the prediction history during warmup.
            For reccurent models, update prediction state with observed positions at the current frame.
        """
        self.mobile_positions[...] = observed_mobile_positions
        self.position_history = np.concatenate(
            [self.position_history, observed_mobile_positions[:, :, None, :]],
            axis=2,
        )
        if self._uses_recurrent_state:
            batch_size, mobile_count = self.mobile_positions.shape[:2]
            positions = torch.as_tensor(
                observed_mobile_positions,
                dtype=torch.float32,
                device=self.latent_state.device,
            ).reshape(batch_size * mobile_count, 3)
            with torch.inference_mode():
                self.latent_state = self.model.encode_predict_update_helper(
                    self.latent_state,
                    positions,
                    self._raw_scene_features,
                    # whether need to predict first before updating - for sequencing
                    advance=not self._state_needs_observation_update, 
                )
            self._observed_frame_count += 1
            self._state_needs_observation_update = False

    def step(self):
        """Advance each predicted mobile by one model step.

        ``mobile_positions`` is a (B, K, 3) array containing all environments
        and mobiles. Static scene features are passed separately when present.
        
        position_history adds a T time dimension to it; latent state models update the latent state

        """
        if self._uses_recurrent_state:
            batch_size, mobile_count = self.mobile_positions.shape[:2]
            with torch.inference_mode():
                predicted_positions, self.latent_state = self.model(
                    self.latent_state
                )
            predicted_positions = predicted_positions.reshape(
                batch_size,
                mobile_count,
                3,
            ).cpu().numpy()
            self._state_needs_observation_update = True
        else:

            required_history_size = self.model.get_required_history()

            desired_history = self.position_history[:, :, -required_history_size:, :]

            predicted_positions = self.model.predict(
                desired_history,
                self.scene_features,
            )

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


class TrajectoryRecorder:
    """Capture translucent snapshots for actual, observed, and predicted mobiles."""

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
        stationary_camera,
        actual_mobiles,
        actual_mobile_configs,
        predicted_mobile_configs,
        ghost_count,
    ):
        self.scene = scene
        self.actual_mobiles = actual_mobiles
        self.stationary_camera = stationary_camera
        self.actual_mobile_configs = {config.name: config for config in actual_mobile_configs}
        self.predicted_mobile_configs = {config.name: config for config in predicted_mobile_configs}
        self.predicted_mobile_names = [config.name for config in predicted_mobile_configs]
        self.ghost_count = ghost_count
        self._next_ghost = 0
        self._next_observation_ghost = 0
        self.ghosts = self._add_hidden_group((0.2, 0.6, 1.0, 0.28), 
                                      actual_mobile_configs)
        self.observation_ghosts = self._add_hidden_group((1.0, 0.85, 0.1, 0.28),
                                                  predicted_mobile_configs)
        self.prediction_ghosts = self._add_hidden_group((1.0, 0.35, 0.1, 0.28),
                                                 predicted_mobile_configs)

    def set_prediction_lateral_offset(self, prediction_lateral_offset):
        """Set the display offset applied to predicted trajectory ghosts."""
        self.prediction_lateral_offset = np.asarray(
            prediction_lateral_offset,
            dtype=float,
        )

    def set_up_predictions(self, predicted_mobiles):
        """Attach the predicted mobile positions used during capture."""
        self.predicted_mobiles = predicted_mobiles

    def capture_observation(self, observed_mobile_positions):
        """Place one laterally offset ghost for every observed mobile."""
        for index, name in enumerate(self.predicted_mobile_names):
            position = observed_mobile_positions[:, index, :]
            self.observation_ghosts[name][self._next_observation_ghost].set_pos(
                position + self.prediction_lateral_offset
            )

        self._next_observation_ghost += 1

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

    def render_image(self):
        """Render the trajectory from the stationary camera.

        The stationary camera is independent of interactive viewer movement,
        so the exported trajectory view remains fixed after scene setup.

        Returns:
            RGB image as a uint8 NumPy array with shape (height, width, 3).
        """
        rendered = self.stationary_camera.render(rgb=True)
        rgb = rendered[0] if isinstance(rendered, tuple) else rendered
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        return rgb.astype("uint8")

    def save_image(self, path, rgb_image=None):
        """Save a trajectory image to disk.

        Args:
            path: Output path accepted by pathlib.Path.
            rgb_image: Optional uint8 RGB array with shape (height, width, 3).
                When omitted, the stationary camera renders the image first.

        Returns:
            Path to the saved image.
        """
        if rgb_image is None:
            rgb_image = self.render_image()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_image).save(output_path)
        return output_path
