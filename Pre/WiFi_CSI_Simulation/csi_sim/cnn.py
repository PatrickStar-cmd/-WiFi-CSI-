"""Optional PyTorch backend: 7,121-parameter shared per-link TinyCNN."""
from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .utils import write_csv


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(56, 16, 5, padding=2), nn.ReLU(),
                                     nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


class WindowDataset(Dataset):
    def __init__(self, refs, labels, mean, std):
        self.refs, self.labels = refs, labels
        self.mean, self.std = mean[:, None], std[:, None]
        self.files = {}

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        path, row, link = self.refs[index]
        if path not in self.files:
            self.files[path] = h5py.File(path, "r")
        x = self.files[path]["windows/amplitude"][row, link]
        return torch.from_numpy(((x - self.mean) / self.std).astype(np.float32)), torch.tensor(self.labels[index], dtype=torch.float32)

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()


def normalization(paths):
    total, square, count = np.zeros(56), np.zeros(56), 0
    for path in paths:
        with h5py.File(path, "r") as f:
            fraction = f["truth/motion_fraction"][...]
            pure = (fraction <= 0.2 + 1e-9) | (fraction >= 0.8 - 1e-9)
            valid = f["windows/valid"][...] & pure[:, None]
            for start in range(0, len(pure), 32):
                x = f["windows/amplitude"][start:start + 32][valid[start:start + 32]].astype(np.float64)
                total += x.sum(axis=(0, 2))
                square += (x ** 2).sum(axis=(0, 2))
                count += x.shape[0] * x.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 1e-10))
    return mean.astype(np.float32), std.astype(np.float32)


def train_cnn(train_examples, val_examples, train_paths, cfg, seed, destination):
    settings = cfg["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(settings["device"])
    model = TinyCNN().to(device)
    assert sum(p.numel() for p in model.parameters()) == 7121
    mean, std = normalization(train_paths)
    _, ty, tr = train_examples
    _, vy, vr = val_examples
    train_set, val_set = WindowDataset(tr, ty, mean, std), WindowDataset(vr, vy, mean, std)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=settings["batch_size"], shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["cnn_lr"])
    criterion = nn.BCEWithLogitsLoss()
    best_loss, stale, best_state, history = float("inf"), 0, None, []
    try:
        for epoch in range(settings["epochs"]):
            model.train()
            train_total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_total += float(loss.detach()) * len(y)
            model.eval()
            val_total = 0.0
            with torch.inference_mode():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_total += float(criterion(model(x), y)) * len(y)
            val_loss = val_total / len(val_set)
            history.append({"epoch": epoch + 1, "train_loss": train_total / len(train_set), "validation_loss": val_loss, "seed": seed})
            print(f"  B3 seed={seed} epoch={epoch + 1} validation_loss={val_loss:.5f}", flush=True)
            if val_loss < best_loss - 1e-7:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= settings["patience"]:
                    break
    finally:
        train_set.close()
        val_set.close()
    if best_state is None:
        raise RuntimeError("CNN validation produced no finite checkpoint.")
    destination = Path(destination)
    torch.save({"state_dict": best_state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                "seed": seed, "validation_loss": best_loss, "parameter_count": 7121}, destination)
    write_csv(destination.with_suffix(".history.csv"), history)
    model.load_state_dict(best_state)
    return {"network": model, "mean": mean, "std": std, "device": device, "validation_loss": best_loss, "seed": seed}


def predict_cnn(artifact, path, valid, batch_size):
    result = np.full(valid.shape, np.nan)
    model = artifact["network"]
    model.eval()
    with h5py.File(path, "r") as f, torch.inference_mode():
        for start in range(0, len(valid), 32):
            mask = valid[start:start + 32]
            x = f["windows/amplitude"][start:start + 32][mask]
            scores = []
            for offset in range(0, len(x), batch_size):
                z = (x[offset:offset + batch_size] - artifact["mean"][None, :, None]) / artifact["std"][None, :, None]
                logits = model(torch.from_numpy(z.astype(np.float32)).to(artifact["device"]))
                scores.extend(np.logaddexp(0.0, logits.cpu().numpy().astype(np.float64)))
            result[start:start + 32][mask] = scores
    return result


def load_cnn(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"network": model, "mean": checkpoint["mean"].numpy(), "std": checkpoint["std"].numpy(),
            "device": torch.device(device), "seed": checkpoint["seed"]}
