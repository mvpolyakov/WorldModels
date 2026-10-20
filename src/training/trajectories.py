from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


from sim.build_world import WorldBuilder

#MIN_MOVEMENT = 0.001 # min object displacement to keep collecting trajectory
MIN_MOVEMENT = 0 # actually do want to have samples of near static objects - this pretty much eliminates it
TRAJECTORIES_PER_SCENE = 4


def _round_up_trajectory_count(n_trajectories: int) -> int:
    """Round a requested trajectory count up to a complete scene group."""

    # round up to a multiple of TRAJECTORIES_PER_SCENE * 6 (5 training per 1 validation hardcoded below)
    return (n_trajectories // (6 * TRAJECTORIES_PER_SCENE) + 1) * 6 * TRAJECTORIES_PER_SCENE


def _get_training_trajectory_count(n_trajectories: int) -> int:
    """Choose a split boundary that keeps scene groups intact when possible."""
    scene_count = n_trajectories // TRAJECTORIES_PER_SCENE

    if scene_count == 1:
        return n_trajectories - 1
    
    validation_scene_count = max(1, scene_count // 5)
    return (scene_count - validation_scene_count) * TRAJECTORIES_PER_SCENE

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
        trajectories_per_scene: int = 1,
        *,
        use_surface_features: bool = False,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Collect trajectories with static pose or top-plane/radius features."""
        if steps < 5:
            raise ValueError("Collect at least 5 simulator steps.")

        world, _ = self.world_builder.build(
            show_viewer=False,
            n_environments=n_environments,
            ghost_count=0,
            random_generator=random_generator,
            trajectories_per_scene=trajectories_per_scene,
        )

        if use_surface_features:
            scene_features = self.world_builder.get_surface_features(world)
        else:
            scene_features = world.scene_features

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
        return trajectories, scene_features.astype(np.float32)


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


class TrajectorySliceDataset(Dataset):
    """Expose resampled fixed-width slices of complete source trajectories.

    New scheme:
    Collect complete trajectories, then train on slices beginning at different times. 
    Reset the latent at each slice's start; do not process earlier observations 
    or provide absolute frame indices.  Keep some slices starting at frame zero. 

    An initial policy is four slices per trajectory per epoch: one beginning at zero 
    and three at randomly chosen later frames. Use a maximum length of [128] frames and
    retain three warm-up observations initially. Resample training starts each epoch.

    Split source trajectories before sampling slices to prevent leakage. Fix validation 
    starts for repeatable comparisons. Short slices remain usable for the targets 
    they contain; exclude padding from state updates and losses. Within each slice, 
    retain the current observation traversal and independent forecast branches after warm-up.

    This removes a reliable clock measured from simulation start. The opposing-ramp task 
    supplies the additional reason to distinguish different histories at similar positions.

    Each source trajectory contributes exactly one slice beginning at frame zero
    and the remaining slices begin at sampled later frames. Slice starts are
    local reset points: no preceding observation or absolute source-frame index
    is included. Samples are padded to ``max_slice_length``; a True entry in the
    padding mask marks a padded frame.


    @param source_dataset: Complete source trajectories and their scene features.
    @param slices_per_trajectory: Total slices produced per source trajectory.
    @param max_slice_length: Fixed padded width and maximum real slice length.
    @param warm_up_length: Observation count required before forecasting. A
        sampled slice must contain at least one target after this warm-up.
    @param sampling_seed: Seed used to derive reproducible epoch-specific starts.
    """

    source_dataset: TrajectorySequenceDataset

    @staticmethod
    def _validate_init_values(
        source_dataset: TrajectorySequenceDataset,
        slices_per_trajectory: int,
        max_slice_length: int,
        warm_up_length: int,
        sampling_seed: int,
    ) -> None:
        """Validate that source trajectories can provide the requested slices."""
        if not isinstance(source_dataset, TrajectorySequenceDataset):
            raise TypeError("source_dataset must be a TrajectorySequenceDataset.")
        if slices_per_trajectory < 1:
            raise ValueError("slices_per_trajectory must be positive.")
        if warm_up_length < 1:
            raise ValueError("warm_up_length must be positive.")

        minimum_slice_length = warm_up_length + 1
        if max_slice_length < minimum_slice_length:
            raise ValueError(
                "max_slice_length must include the warm-up and one target frame."
            )
        if sampling_seed < 0:
            raise ValueError("sampling_seed must be nonnegative.")

        minimum_source_length = warm_up_length + slices_per_trajectory
        too_short_indices = [
            trajectory_index
            for trajectory_index, trajectory in enumerate(source_dataset.trajectories)
            if len(trajectory) < minimum_source_length
        ]
        if too_short_indices:
            raise ValueError(
                "Source trajectories do not have enough distinct valid slice starts: "
                f"{too_short_indices}."
            )

    def __init__(
        self,
        source_dataset: TrajectorySequenceDataset,
        *,
        slices_per_trajectory: int,
        max_slice_length: int,
        warm_up_length: int,
        sampling_seed: int,
    ):
        self._validate_init_values(
            source_dataset,
            slices_per_trajectory,
            max_slice_length,
            warm_up_length,
            sampling_seed,
        )

        self.source_dataset = source_dataset
        self.slices_per_trajectory = slices_per_trajectory
        self.max_slice_length = max_slice_length
        self.warm_up_length = warm_up_length
        self.sampling_seed = sampling_seed

        self._slice_metadata = np.empty((0, 3), dtype=np.int64)
        self.resample_slices(epoch=0)

    def resample_slices(self, epoch: int) -> None:
        """Re-slice trajectories from stored dataset into slices - precompute 
        slice indices so that __getitem__ just serves views into the slices one at a time.        

        The one frame-zero slice per trajectory remains fixed. Validation keeps
        its constructor-created starts by not calling this method.

        @param epoch: Zero-based training epoch used with the sampling seed.
        """
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")

        # Deriving the generator each time (with the epoch) makes 
        # repeated or resumed calls stable.
        random_generator = np.random.default_rng([self.sampling_seed, epoch])

        minimum_slice_length = self.warm_up_length + 1
        slice_metadata = []

        for trajectory_index, trajectory in enumerate(
            self.source_dataset.trajectories
        ):
            slice_metadata.append(
                (
                    trajectory_index,
                    0,
                    min(self.max_slice_length, len(trajectory)),
                )
            )

            later_slice_count = self.slices_per_trajectory - 1
            if later_slice_count == 0:
                continue

            latest_start_frame = len(trajectory) - minimum_slice_length
            later_start_frames = random_generator.choice(
                np.arange(1, latest_start_frame + 1),
                size=later_slice_count,
                replace=False,
            )
            for start_frame in later_start_frames:
                slice_metadata.append(
                    (
                        trajectory_index,
                        int(start_frame),
                        min(
                            self.max_slice_length,
                            len(trajectory) - int(start_frame),
                        ),
                    )
                )

        self._slice_metadata = np.asarray(slice_metadata, dtype=np.int64)

    def get_slice_metadata(self) -> np.ndarray:
        """Return source trajectory, start frame, and length for every slice.

        @return: Integer array shaped (S, 3), with columns for source trajectory
            index, inclusive source start frame, and unpadded slice length.
        """
        return self._slice_metadata.copy()

    def __len__(self) -> int:
        """Return the current number of trajectory slices."""
        return len(self._slice_metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return one fixed-width slice and its source scene information.

        @param index: Index into the current epoch's slice specifications.

        @return: Positions (M, 3), scene features (F,), scalar unpadded length,
            and padding_frame_mask (M,). M is max_slice_length; True marks
            padding and False marks an observation.
        """
        trajectory_index, start_frame, slice_length = (
            int(value) for value in self._slice_metadata[index]
        )

        positions = torch.from_numpy(
            self.source_dataset.trajectories[trajectory_index][
                start_frame:start_frame + slice_length
            ]
        )
        
        if slice_length < self.max_slice_length:
            padded_positions = torch.zeros(
                (self.max_slice_length, 3),
                dtype=positions.dtype,
                device="cpu",
            )
            padded_positions[:slice_length] = positions
            positions = padded_positions

        padding_frame_mask = torch.zeros(
            self.max_slice_length,
            dtype=torch.bool,
            device="cpu",
        )
        padding_frame_mask[slice_length:] = True
        return (
            positions,
            self.source_dataset.scene_feature_tensor[trajectory_index],
            torch.tensor(slice_length, dtype=torch.long, device="cpu"),
            padding_frame_mask,
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

    n_trajectories = _round_up_trajectory_count(n_trajectories)

    random_generator = np.random.default_rng(None if seed == 0 else seed)
    trajectories, scene_features = SimulationTrajectoryCollector(
        world_builder
    ).collect_randomized_batch(
        steps,
        # 1 trajectory per environment
        n_environments=n_trajectories,
        random_generator=random_generator,
        trajectories_per_scene=TRAJECTORIES_PER_SCENE,
    )
    # computes the train/validation split too
    training_trajectory_count = _get_training_trajectory_count(n_trajectories)

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

    Approximately 20 percent of complete scenes are reserved for validation,
    without dividing four-trajectory scene groups when multiple scenes exist.
    Each split pads its trajectories once to its own maximum length.

    @param world_builder: Builder used to create the randomized simulation batch.
    @param seed: Randomization seed; zero requests nondeterministic collection.
    @param steps: Maximum simulator steps collected per trajectory.
    @param n_trajectories: Requested trajectory count, rounded up to a multiple
        of four.

    @return: All raw position arrays, the training TrajectorySequenceDataset,
        and the validation TrajectorySequenceDataset.
    """
    n_trajectories = _round_up_trajectory_count(n_trajectories)

    random_generator = np.random.default_rng(None if seed == 0 else seed)

    training_trajectory_count = _get_training_trajectory_count(n_trajectories)

    training_trajectories, training_scene_features = SimulationTrajectoryCollector(
        world_builder
    ).collect_randomized_batch(
        steps,
        n_environments=training_trajectory_count,
        random_generator=random_generator,
        trajectories_per_scene=TRAJECTORIES_PER_SCENE,
        use_surface_features=True,
    )

    training_dataset = TrajectorySequenceDataset(
        training_trajectories,
        training_scene_features,
    )

    validation_trajectories, validation_scene_features = SimulationTrajectoryCollector(
        world_builder
    ).collect_randomized_batch(
        steps,
        n_environments=n_trajectories - training_trajectory_count,
        random_generator=random_generator,
        trajectories_per_scene=TRAJECTORIES_PER_SCENE,
        use_surface_features=True,
    )

    validation_dataset = TrajectorySequenceDataset(
        validation_trajectories,
        validation_scene_features,
    )

    print('\ntrajectories used:', len(training_trajectories), '/', len(validation_trajectories))

    # trajectories themselves are only returned here for saving/logging
    return training_trajectories + validation_trajectories, training_dataset, validation_dataset
