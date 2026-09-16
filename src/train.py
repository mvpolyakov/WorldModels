from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from sim.build_world import load_world_builder
from training import (
    PositionHistoryDataset,
    SimulationTrajectoryCollector,
    create_trajectory_dataset
)
from world_models import PositionHistoryMLP

HISTORY_WINDOW = 5
PREDICTION_HORIZON = 5

def _process_epoch_pass(model, data_loader, device, optimizer):
    total_loss = 0.0
    sample_count = 0

    for batch in data_loader:
        histories, targets, scene_features = batch
        histories = histories.to(device)
        targets = targets.to(device)
        scene_features = scene_features.to(device)
        if optimizer is not None:
            optimizer.zero_grad()

        # Predict multiple steps into the future progressively adding predictions into
        # history for more informative loss calc
        predictions = []
        rollout_history = histories
        for _ in range(targets.shape[1]):
            predicted_position = model(rollout_history, scene_features)
            predictions.append(predicted_position)
            rollout_history = torch.cat(
                [rollout_history[:, 1:], predicted_position.unsqueeze(1)],
                dim=1,
            )
        predictions = torch.stack(predictions, dim=1)
        loss = torch.mean((predictions - targets) ** 2)
        if not torch.isfinite(loss):
            error = ('Validation' if optimizer is None else 'Training') + "loss became non-finite."
            raise RuntimeError(error)
        
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        count = len(histories)
        total_loss += loss.item() * count
        sample_count += count

    return total_loss / sample_count

def train_model(model,
                train_data_loader,
                validation_data_loader,
                config,
                device):

    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    training_errors = []
    validation_errors = []

    for epoch in range(config['epochs']):
        model.train()

        mean_error =_process_epoch_pass(model, 
                                        train_data_loader, 
                                        device, 
                                        optimizer)

        training_errors.append(mean_error)

        model.eval()
        with torch.inference_mode():
            mean_error =_process_epoch_pass(model, 
                                            validation_data_loader, 
                                            device, 
                                            None)

            validation_errors.append(mean_error)

        if epoch % 2 == 0:
            print   ('epoch', epoch)

    return training_errors, validation_errors

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a phase 1.0 position predictor.")
    parser.add_argument("--world-builder", default="simple-rolling")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
   # parser.add_argument("--output-dir", default="artifacts/phase1_0/run_001")
    return parser.parse_args()

def main() -> None:
    args = get_args()

    import genesis as gs

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        gs.init(backend=gs.gpu, seed=args.seed)
    else:
        gs.init(backend=gs.gpu)

    world_builder = load_world_builder(args.world_builder)

    trajectories, training_dataset, validation_dataset = create_trajectory_dataset(
        world_builder,
        args.seed,
        args.steps,
        args.trajectories,
        HISTORY_WINDOW,
        PREDICTION_HORIZON,
    )

    # Genesis sets PyTorch's default device to its backend during initialization.
    # Keep DataLoader indices and shuffle permutations on CPU.
    torch.set_default_device("cpu")
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PositionHistoryMLP(HISTORY_WINDOW, args.hidden_size).to(device)

    config = {'lr': args.learning_rate,
              'epochs': args.epochs
              }

    # Later move into a dedicated class
    training_errors, validation_errors = train_model(
        model,
        training_loader,
        validation_loader,
        config,
        device,
    )

    output_dir = Path('artifacts/runs')

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_lengths = np.asarray([len(trajectory) for trajectory in trajectories])
    saved_positions = np.full(
        (len(trajectories), trajectory_lengths.max(), 3),
        np.nan,
        dtype=np.float32,
    )
    for trajectory_index, trajectory in enumerate(trajectories):
        saved_positions[trajectory_index, :len(trajectory)] = trajectory
    np.savez(
        output_dir / "trajectory.npz",
        positions=saved_positions,
        trajectory_lengths=trajectory_lengths,
        scene_features=np.stack(
            [
                trajectory_dataset.scene_features
                for trajectory_dataset in (
                    training_dataset.datasets + validation_dataset.datasets
                )
            ]
        ),
    )
    torch.save(
        {
            "model_type": "PositionHistoryMLP",
            "history_length": HISTORY_WINDOW,
            "prediction_horizon": PREDICTION_HORIZON,
            "hidden_size": args.hidden_size,
            "scene_feature_size": model.scene_feature_size,
            "state_dict": model.state_dict(),
        },
        output_dir / "checkpoint.tar",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "training_error": training_errors,
                "validation_error": validation_errors,
            },
            indent=2,
        )
    )
    print(f"Saved phase 1.0 artifacts to {output_dir}")


if __name__ == "__main__":
    main()
