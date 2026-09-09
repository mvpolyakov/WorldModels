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

def train_model(model, 
                data_loader, 
                config,
                device):
    
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    losses = []

    for epoch in range(config['epochs']):
        model.train()

        total_loss = 0.0
        sample_count = 0

        for histories, targets in data_loader:
            histories = histories.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(histories)
            loss = torch.mean((predictions - targets) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite.")
            loss.backward()
            optimizer.step()
            count = len(histories)
            total_loss += loss.item() * count
            sample_count += count

        losses.append(total_loss / sample_count)

        if epoch % 10 == 0:
            print   ('epoch', epoch)

    # Once training structure changes and it makes sense, may be add an overall evaluation here?
    #model.eval()
    #with torch.inference_mode():
    #  ...

    return losses

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

    trajectories, dataset = create_trajectory_dataset(world_builder,
                                                        args.seed, 
                                                        args.steps,
                                                        args.trajectories,
                                                        HISTORY_WINDOW)

    # Genesis sets PyTorch's default device to its backend during initialization.
    # Keep DataLoader indices and shuffle permutations on CPU.
    torch.set_default_device("cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = PositionHistoryMLP(HISTORY_WINDOW, args.hidden_size).to(device)

    config = {'lr': args.learning_rate,
              'epochs': args.epochs
              }

    # Later move into a dedicated class
    losses = train_model(model, loader, config, device)

    output_dir = Path('artifacts/runs')

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "trajectory.npz", positions=np.stack(trajectories))
    torch.save(
        {
            "model_type": "PositionHistoryMLP",
            "history_length": HISTORY_WINDOW,
            "hidden_size": args.hidden_size,
            "state_dict": model.state_dict(),
        },
        output_dir / "checkpoint.tar",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps({"epochs": losses}, indent=2)
    )
    print(f"Saved phase 1.0 artifacts to {output_dir}")


if __name__ == "__main__":
    main()
