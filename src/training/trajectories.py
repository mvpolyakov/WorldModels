from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


from sim.build_world import WorldBuilder

#MIN_MOVEMENT = 0.001 # min object displacement to keep collecting trajectory
MIN_MOVEMENT = 0 # actually do want to have samples of near static objects - this pretty much eliminates it

class SimulationTrajectoryCollector:
    """Collect ball-position trajectories from a batched Genesis world.

        WILL NEED TO UPDATE THIS FOR ANY MOBILES CHANGES
    """

    def __init__(self, world_builder: WorldBuilder):
        if len(world_builder.config.mobiles) != 1:
            raise ValueError("Phase 1.0 requires exactly one mobile object.")

        self.world_builder = world_builder

    def collect_randomized_batch(
        self,
        steps: int,
        n_environments: int,
        random_generator: np.random.Generator,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Collect trajectories and their static ramp/radius features."""
        if steps < 5:
            raise ValueError("Collect at least 5 simulator steps.")

        world, _ = self.world_builder.build(
            show_viewer=False,
            n_environments=n_environments,
            ghost_count=0,
            random_generator=random_generator,
        )

        positions_by_step = [
            world.mobiles["ball"].get_pos().detach().cpu().numpy().copy()
        ]

        stationary_frame_counts = np.zeros(n_environments, dtype=np.int32)
        trajectory_end_indices = np.full(n_environments, -1, dtype=np.int32)

        for _ in range(steps):
            world.step()

            # All of this just keeps track of when each individual trajectory seizes substantial movement for an early cutoff 
            # (don't want a bunch of samples of essentially stationary objects)

            current_positions = (
                world.mobiles["ball"].get_pos().detach().cpu().numpy().copy()
            )
            displacements = np.linalg.norm(
                current_positions - positions_by_step[-1],
                axis=1,
            )
            stationary_frame_counts = np.where(
                displacements < MIN_MOVEMENT,
                stationary_frame_counts + 1,
                0,
            )
            positions_by_step.append(current_positions)
            newly_stationary = (
                (stationary_frame_counts >= 5) & (trajectory_end_indices < 0)
            )
            trajectory_end_indices[newly_stationary] = len(positions_by_step)

        positions = np.stack(positions_by_step)

        # have to collect all trajectories fully but then cut off individual trajectories where movement is too small 
        # they will be seamlessly combined regardless of length in the dataset collection later
        trajectory_end_indices[trajectory_end_indices < 0] = len(positions_by_step)
        trajectories = [
            positions[:trajectory_end_indices[environment_index], environment_index, :]
            for environment_index in range(n_environments)
        ]
        return trajectories, world.scene_features.astype(np.float32)


class PositionHistoryDataset(Dataset):
    """Create position histories and their subsequent rollout targets."""

    def __init__(
        self,
        positions: np.ndarray,
        history_length: int = 5,
        prediction_horizon: int = 5,
        *,
        scene_features: np.ndarray,
    ):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (frames, 3)")
        if len(positions) < history_length + prediction_horizon:
            raise ValueError("Need enough frames for the history and prediction horizon.")
        scene_features = np.asarray(scene_features, dtype=np.float32)
        if scene_features.ndim != 1:
            raise ValueError("scene_features must have shape (features,)")
        self.positions = positions
        self.scene_features = scene_features

        sample_count = len(positions) - history_length - prediction_horizon + 1
        self.X = np.stack(
            [positions[i : i + history_length] for i in range(sample_count)]
        )
        self.targets = np.stack(
            [
                positions[i + history_length : i + history_length + prediction_horizon]
                for i in range(sample_count)
            ]
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = (
            torch.from_numpy(self.X[index]),
            torch.from_numpy(self.targets[index]),
        )
        return sample + (torch.from_numpy(self.scene_features),)


class TrajectorySequenceDataset(Dataset):
    """Expose each complete trajectory as one latent-state training sample.

    Pad positions once to the longest trajectory in this dataset so the standard
    DataLoader can stack samples. N is trajectory count, T is the maximum frame
    count, and F is scene-feature width.

    @param trajectories: List of float arrays shaped (T_i, 3), where T_i is
        each trajectory's original frame count.
    @param scene_features: Float array shaped (N, F), one row per trajectory.
    """

    def __init__(
        self,
        trajectories: list[np.ndarray],
        scene_features: np.ndarray,
    ):
        """Validate trajectories and prepare padded CPU tensors.

        Pads all trajectories in the dataset to same length (max).
        Dataset-wide padding keeps batching simple, but costs storage and network
        computation for the extra padding in every batch; masking only removes its losses.
        If lengths vary substantially, an alternative is to keep variable-length
        tensors and use a custom collate function to pad each batch to its own
        longest trajectory.
        """
        if len(trajectories) == 0:
            raise ValueError("At least one trajectory is required.")
        
        #features = np.asarray(scene_features, dtype=np.float32)
        if scene_features.ndim != 2 or len(scene_features) != len(trajectories):
            raise ValueError(
                "scene_features must have shape (trajectories, features)"
            )

        self.trajectories = []
        for trajectory in trajectories:
            #trajectory = np.asarray(trajectory, dtype=np.float32) declared as such
            if trajectory.ndim != 2 or trajectory.shape[1] != 3:
                raise ValueError("Each trajectory must have shape (frames, 3)")
            if len(trajectory) == 0:
                raise ValueError("Trajectories must contain at least one frame.")
            self.trajectories.append(trajectory)

        # Retain unpadded arrays for rollout evaluation and artifact export.
        self.padded_positions = pad_sequence(
            [torch.from_numpy(trajectory) for trajectory in self.trajectories],
            batch_first=True,
        )
        
        # keep data on cpu and move to device on each batch processing step
        # with large data this is a requirement with small (here) not much gain from alternative
        self.scene_feature_tensor = torch.from_numpy(scene_features)

        # Dataset creation can follow Genesis initialization, which changes the
        # default device. Keep stored data and masks on CPU until training.
        self.trajectory_lengths = torch.tensor(
            [len(trajectory) for trajectory in self.trajectories],
            dtype=torch.long,
            device="cpu",
        )
        frame_indices = torch.arange(self.padded_positions.shape[1], device="cpu")
        self.padding_frame_mask = (
            frame_indices.unsqueeze(0) >= self.trajectory_lengths.unsqueeze(1)
        )

    def __len__(self) -> int:
        """Return the number of complete trajectories."""
        return len(self.trajectories)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return a padded trajectory and its original length for batching.

        @param index: Trajectory index.

        @return: Positions (T, 3), scene features (F,), scalar trajectory length,
            and padding_frame_mask (T,). True marks trailing padding and False
            marks an observation; T is the dataset's maximum trajectory length.
        """
        return (
            self.padded_positions[index],
            self.scene_feature_tensor[index],
            self.trajectory_lengths[index],
            self.padding_frame_mask[index],
        )


def create_trajectory_dataset(world_builder,
                              seed, 
                              steps,
                              n_trajectories,
                              required_history,
                              prediction_horizon=5):
    '''
    create n_trajectories randomized rollouts for teaching a model basic rolling physics
    '''

    if n_trajectories < 2:
        raise ValueError("At least 2 trajectories are required for training and validation.")

    random_generator = np.random.default_rng(None if seed == 0 else seed)
    trajectories, scene_features = SimulationTrajectoryCollector(
        world_builder
    ).collect_randomized_batch(
        steps,
        # 1 trajectory per environment
        n_environments=n_trajectories,
        random_generator=random_generator,
    )
    validation_trajectory_count = max(1, n_trajectories // 5)
    training_trajectory_count = n_trajectories - validation_trajectory_count

    training_trajectory_datasets = [
        PositionHistoryDataset(
            positions,
            required_history,
            prediction_horizon,
            scene_features=features,
        )
        for positions, features in zip(
            trajectories[:training_trajectory_count],
            scene_features[:training_trajectory_count],
        )
    ]
    validation_trajectory_datasets = [
        PositionHistoryDataset(
            positions,
            required_history,
            prediction_horizon,
            scene_features=features,
        )
        for positions, features in zip(
            trajectories[training_trajectory_count:],
            scene_features[training_trajectory_count:],
        )
    ]

    return (
        trajectories,
        ConcatDataset(training_trajectory_datasets),
        ConcatDataset(validation_trajectory_datasets),
    )


def create_trajectory_sequence_dataset(
    world_builder,
    seed,
    steps,
    n_trajectories,
):
    """Collect simulator trajectories and split them without splitting sequences.

    20 percent of trajectories (at least one), are reserved for
    validation. Each split pads its trajectories once to its own maximum length.

    @param world_builder: Builder used to create the randomized simulation batch.
    @param seed: Randomization seed; zero requests nondeterministic collection.
    @param steps: Maximum simulator steps collected per trajectory.
    @param n_trajectories: Number of randomized environments to collect.

    @return: All raw position arrays, the training TrajectorySequenceDataset,
        and the validation TrajectorySequenceDataset.
    """
    if n_trajectories < 2:
        raise ValueError("At least 2 trajectories are required for training and validation.")

    random_generator = np.random.default_rng(None if seed == 0 else seed)
    trajectories, scene_features = SimulationTrajectoryCollector(
        world_builder
    ).collect_randomized_batch(
        steps,
        n_environments=n_trajectories,
        random_generator=random_generator,
    )
    validation_trajectory_count = max(1, n_trajectories // 5)
    training_trajectory_count = n_trajectories - validation_trajectory_count

    training_dataset = TrajectorySequenceDataset(
        trajectories[:training_trajectory_count],
        scene_features[:training_trajectory_count],
    )
    validation_dataset = TrajectorySequenceDataset(
        trajectories[training_trajectory_count:],
        scene_features[training_trajectory_count:],
    )
    return trajectories, training_dataset, validation_dataset
