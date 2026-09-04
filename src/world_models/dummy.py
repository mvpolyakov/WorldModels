import numpy as np


class DummyWorldModel:
    """Generate a small random-walk prediction for a batch of positions."""

    def __init__(self, seed=0, offset_scale=0.03):
        self._random = np.random.default_rng(seed)
        self.offset_scale = offset_scale

    def predict(self, current_positions: np.ndarray) -> np.ndarray:
        """Predict a batch of all mobile positions.

        Args:
            current_positions: Array with shape (B, K, 3), where B is the
                number of environments, K is the number of mobiles, and 3
                contains XYZ coordinates.

        Returns:
            Array with the same (B, K, 3) shape as the input.
        """
        predicted_positions = []
        for environment_positions in current_positions:
            environment_predictions = []
            for position in environment_positions:
                start = np.asarray(position, dtype=float)
                offset = self._random.normal(0.0, self.offset_scale, size=3)
                environment_predictions.append(start + offset)
            predicted_positions.append(environment_predictions)

        return np.asarray(predicted_positions)
