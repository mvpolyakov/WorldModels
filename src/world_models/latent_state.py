import torch
from torch import nn

from sim.scene_features import OPPOSING_RAMPS_SCENE_FEATURE_SIZE


def _build_mlp(input_size: int, output_size: int, hidden_size: int) -> nn.Sequential:
    """Build an MLP with two ReLU-activated hidden layers.

    @param input_size: Number of input values per sample.
    @param output_size: Number of output values per sample.
    @param hidden_size: Width of both hidden layers.

    @return: Sequential linear network mapping input_size to output_size.
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )

def _build_mlp2(input_size: int, output_size: int) -> nn.Sequential:
    """Build an MLP with two ReLU-activated hidden layers.

    @param input_size: Number of input values per sample.
    @param output_size: Number of output values per sample.
    @param hidden_size: Width of both hidden layers.

    @return: Sequential linear network mapping input_size to output_size.
    """

    hidden_size = 64
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class LatentStateModel(nn.Module):
    """Encode observations and predict a caller-owned latent state through time.

    B is batch size, L is latent size, E is observation embedding size, and F
    is scene-feature size. The model
    operates on one object per row; it never stores a row's current state.

    @param latent_size: L, the persistent latent width.
    @param observation_embedding_size: E, the encoded observation width.
    @param hidden_size: Width of each MLP's two hidden layers.
    @param scene_feature_size: F, the static scene-context width.
    @param warm_up_length: Number of observed frames before autonomous prediction.
    """

    def __init__(
        self,
        latent_size: int = 64,
        hidden_size: int = 64,
        scene_feature_size: int = OPPOSING_RAMPS_SCENE_FEATURE_SIZE,
        warm_up_length: int = 5,
        *,
        observation_embedding_size: int,
    ):
        """Configure the observation, update, transition, and decoder MLPs."""
        super().__init__()
        if latent_size < 1 or hidden_size < 1:
            raise ValueError("latent_size and hidden_size must be positive.")
        if observation_embedding_size < 1:
            raise ValueError("observation_embedding_size must be positive.")
        if scene_feature_size < 1 or warm_up_length < 1:
            raise ValueError("scene_feature_size and warm_up_length must be positive.")

        self.latent_size = latent_size
        self.observation_embedding_size = observation_embedding_size
        self.hidden_size = hidden_size
        self.scene_feature_size = scene_feature_size
        self.warm_up_length = warm_up_length

        # Component models
        self.observation_encoder = _build_mlp(
            3 + scene_feature_size,                 # 3=position of a single object
            observation_embedding_size,
            hidden_size,
        )
        self.state_updater = _build_mlp(
            latent_size + observation_embedding_size,
            latent_size,
            hidden_size,
        )
        self.dynamics_model = _build_mlp2(latent_size, latent_size)
        
        self.latent_state_decoder = _build_mlp(latent_size, 3, hidden_size)

    def get_required_history(self) -> int:
        """Return the number of observations required before prediction."""
        return self.warm_up_length

    def initialize_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create the caller-owned zero latent used before the first observation.

        @param batch_size: B, the number of independent object states.
        @param device: Optional output device; defaults to the model's device.
        @param dtype: Optional output dtype; defaults to the model's dtype.

        @return: Zero latent tensor with shape (B, L).
        """
        reference = next(self.parameters())
        return torch.zeros(
            (batch_size, self.latent_size),
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def encode_observation(
        self,
        position: torch.Tensor,
        scene_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode an observed position and scene context for a state update.

        @param position: Float tensor with shape (B, 3), containing XYZ positions.
        @param scene_features: Float tensor with shape (B, F).

        @return: Observation embedding with shape (B, E).
        """
        return self.observation_encoder(torch.cat([position, scene_features], dim=1))

    def update_state(
        self,
        prior_state: torch.Tensor,
        observation_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Combine a frame's prior latent and observation embedding.

        @param prior_state: Predicted latent at the observed frame, shape (B, L).
        @param observation_embedding: Encoded observation, shape (B, E).

        @return: Observation-informed latent for that frame, shape (B, L).
        """
        return self.state_updater(
            torch.cat([prior_state, observation_embedding], dim=1)
        )

    def predict_state(self, state: torch.Tensor) -> torch.Tensor:
        """Predict the next frame's latent with a learned residual update.

        @param state: Current latent with shape (B, L).

        @return: Predicted next-frame latent with shape (B, L).
        """
        return state + self.dynamics_model(state)

    def decode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Decode absolute XYZ positions from latent states.

        @param state: Latent tensor with shape (B, L).

        @return: Absolute XYZ positions with shape (B, 3).
        """
        return self.latent_state_decoder(state)

    def encode_predict_update_helper(
        self,
        state: torch.Tensor,
        position: torch.Tensor,
        scene_features: torch.Tensor,
        *,
        advance: bool,
    ) -> torch.Tensor:
        """Combines the encoding of a raw observation, doing 1 step prediction and then 
        updating the next step prediction with the encoded observation
        
        The first observation updates the zero latent without advancing. Later
        observations normally advance first. A correction to an already-predicted
        frame sets advance to False so that frame is not predicted twice.

        @param state: Latest caller-owned latent with shape (B, L).
        @param position: Observed XYZ positions with shape (B, 3).
        @param scene_features: Static scene context with shape (B, F).
        @param advance: Whether state belongs to the preceding frame. - if yes need to predict before updating

        @return: Observation-informed latent at the observed frame, shape (B, L).
        """
        prior_state = self.predict_state(state) if advance else state
        observation_embedding = self.encode_observation(position, scene_features)
        return self.update_state(prior_state, observation_embedding)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one frame and decode its absolute position.

        @param state: Current latent with shape (B, L).

        @return: Tuple of predicted XYZ positions (B, 3) and next latent (B, L).
        """
        next_state = self.predict_state(state)
        return self.decode_state(next_state), next_state
