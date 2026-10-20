import torch
from torch import nn

from sim.scene_features import (
    compute_surface_gap,
    find_closest_surface,
    unpack_surface_features,
)

DYNAMICS_OUTPUT_WEIGHT_GAIN = .01
DECODER_OUTPUT_WEIGHT_GAIN = .1 


def _initialize_mlp(
    layers: list[nn.Module],
    final_weight_gain: float | None,
) -> None:
    """Zero linear biases and optionally reduce the output-layer weight scale.

    @param layers: Ordered modules that will form an MLP.
    @param final_weight_gain: Xavier gain for the last Linear layer, or None to
        retain its default weight initialization.
    """
    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    for layer in linear_layers:
        nn.init.zeros_(layer.bias)
    if final_weight_gain is not None:
        nn.init.xavier_uniform_(
            linear_layers[-1].weight,
            gain=final_weight_gain,
        )


def _build_mlp(
    input_size: int,
    output_size: int,
    hidden_size: int,
    *,
    layer_norm_enabled: bool,
    final_weight_gain: float | None,
) -> nn.Sequential:
    """Build an MLP with two optionally normalized hidden layers.

    @param input_size: Number of input values per sample.
    @param output_size: Number of output values per sample.
    @param hidden_size: Width of both hidden layers.
    @param layer_norm_enabled: Whether to normalize each hidden linear output.
    @param final_weight_gain: Optional Xavier gain for the output weights.

    @return: Sequential linear network mapping input_size to output_size.
    """
    layers = [nn.Linear(input_size, hidden_size)]
    if layer_norm_enabled:
        layers.append(nn.LayerNorm(hidden_size))
    layers.extend((nn.ReLU(), nn.Linear(hidden_size, hidden_size)))
    if layer_norm_enabled:
        layers.append(nn.LayerNorm(hidden_size))
    layers.extend((nn.ReLU(), nn.Linear(hidden_size, output_size)))
    _initialize_mlp(layers, final_weight_gain)
    return nn.Sequential(*layers)


class ObservationEncoder(nn.Module):
    """Distill raw coordinate observations for latent-state updates.

    The coordinate input path prepares position, closest-surface gap and normal,
    and radius. These values pass directly to the state updater; learned visual
    encoding can replace this simplified path when observations become images.
    B is batch size and S is surface count.
    """

    # XYZ position + gap + normal XYZ + radius. These distilled values currently
    # pass directly to the state updater without a learned encoding.
    scene_feature_size = 8

    def compute_closest_surface_context(
        self,
        positions: torch.Tensor,
        surface_features: torch.Tensor,
    ) -> torch.Tensor:
        """Select a plane and assemble its context for each observation.

        @param positions: Observed ball-center XYZ positions, shape (B, 3).
        @param surface_features: Packed plane records and radius, shape (B, 6*S + 1).

        @return: Signed gap and selected outward normal XYZ, shape (B, 4).
        """
        surface_centers, surface_normals, ball_radii = unpack_surface_features(
            surface_features
        )
        selected_surface_index = find_closest_surface(
            positions,
            surface_centers,
            surface_normals,
        )
        batch_indices = torch.arange(positions.shape[0], device=positions.device)
        selected_surface_center = surface_centers[
            batch_indices,
            selected_surface_index,
        ]
        selected_surface_normal = surface_normals[
            batch_indices,
            selected_surface_index,
        ]
        surface_gap = compute_surface_gap(
            positions,
            selected_surface_center,
            selected_surface_normal,
            ball_radii,
        )
        return torch.cat([surface_gap, selected_surface_normal], dim=1)

    def forward(
        self,
        position: torch.Tensor,
        scene_features: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare distilled coordinate and surface features for a state update.

        @param position: Observed ball-center XYZ positions, shape (B, 3).
        @param scene_features: Packed plane records and radius, shape (B, 6*S + 1).

        @return: Position, gap, normal, and radius, shape (B, 8).
        """
        surface_context = self.compute_closest_surface_context(position, scene_features)
        ball_radii = scene_features[:, -1:]
        return torch.cat([position, surface_context, ball_radii], dim=1)


class LatentStateModel(nn.Module):
    """Encode observations and predict a caller-owned latent state through time.

    B is batch size, L is latent size, and F is packed surface-feature size
    (6*S + 1 for S surfaces). The model operates on one object per row; it never
    stores a row's current state.

    @param latent_size: L, the persistent latent width.
    @param hidden_size: Width of every learned component's hidden layers.
    @param warm_up_length: Number of observed frames before autonomous prediction.
    @param layer_norm_enabled: Whether hidden MLP layers use LayerNorm.
    @param dynamics_output_weight_gain: Initial Xavier gain for the residual output.
    @param decoder_output_weight_gain: Initial Xavier gain for displacement output.
    """

    def __init__(
        self,
        latent_size: int = 64,
        hidden_size: int = 64,
        warm_up_length: int = 5,
        *,
        layer_norm_enabled: bool = True,
        dynamics_output_weight_gain: float = DYNAMICS_OUTPUT_WEIGHT_GAIN,
        decoder_output_weight_gain: float = DECODER_OUTPUT_WEIGHT_GAIN,
    ):
        """Configure observation preparation, update, transition, and decoding."""
        super().__init__()
        if latent_size < 1 or hidden_size < 1:
            raise ValueError("latent_size and hidden_size must be positive.")
        if warm_up_length < 1:
            raise ValueError("warm_up_length must be positive.")

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.warm_up_length = warm_up_length
        self.layer_norm_enabled = layer_norm_enabled
        self.dynamics_output_weight_gain = dynamics_output_weight_gain
        self.decoder_output_weight_gain = decoder_output_weight_gain

        # Component models
        self.observation_encoder = ObservationEncoder()
        self.previous_encoded_observation = None # for diffing

        self.state_updater = _build_mlp(
            latent_size + 2 * self.observation_encoder.scene_feature_size,
            latent_size,
            hidden_size,
            layer_norm_enabled=layer_norm_enabled,
            final_weight_gain=None,
        )
        self.dynamics_model = _build_mlp(
            latent_size,
            latent_size,
            hidden_size,
            layer_norm_enabled=layer_norm_enabled,
            final_weight_gain=dynamics_output_weight_gain,
        )

        self.latent_state_decoder = _build_mlp(
            latent_size,
            3,
            hidden_size,
            layer_norm_enabled=False,
            final_weight_gain=decoder_output_weight_gain,
        )

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
        self.previous_encoded_observation = None
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
        observation_reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode an observed position and scene context for a state update.

        @param position: Float tensor with shape (B, 3), containing XYZ positions.
        @param scene_features: Packed top-plane records and radius, shape (B, F).
        @param observation_reset_mask: Optional Boolean tensor shaped (B,); True
            makes the current observation the first frame of a new episode.

        @return: Observation embedding with shape (B, E).
        """
        encoded_observation = self.observation_encoder(position, scene_features)

        if self.previous_encoded_observation is None:
            self.previous_encoded_observation = encoded_observation
        elif observation_reset_mask is not None:
            self.previous_encoded_observation = torch.where(
                observation_reset_mask.unsqueeze(-1),
                encoded_observation,
                self.previous_encoded_observation,
            )

        encoded_difference = encoded_observation - \
                                self.previous_encoded_observation

        self.previous_encoded_observation = encoded_observation

        return torch.cat([encoded_observation, encoded_difference], dim=1)

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

        B is trajectory batch size, R is forecast-start count, and L is latent
        width. The optional R dimension batches independent rollout starts.

        @param state: Current latent with shape (B, L) or (B, R, L).

        @return: Predicted latent with the same shape as state.
        """
        return state + self.dynamics_model(state)

    def decode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Decode XYZ displacements from latent states.

        B is trajectory batch size, R is forecast-start count, and L is latent
        width.

        @param state: Latent tensor with shape (B, L) or (B, R, L).

        @return: XYZ displacements with shape (B, 3) or (B, R, 3).
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
        """Predict one frame and decode its displacement.

        @param state: Current latent with shape (B, L).

        @return: Tuple of predicted XYZ displacements (B, 3) and next latent
            (B, L). The caller adds the displacement to its current position.
        """
        next_state = self.predict_state(state)
        return self.decode_state(next_state), next_state
