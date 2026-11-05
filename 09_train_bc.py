"""09_train_bc.py
Action-chunking behavior cloning：图像 + 最近 H 帧 qpos -> 未来 K 步动作。

用法（服务器上）：
    python 09_train_bc.py --epochs 30 --batch 64 --device cuda

数据：~/robot-project/data/raw/ep_*.npz
输出：~/robot-project/runs/bc/best.pt, norm_stats.npz
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bc_model import BCNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/lqs/robot-project/data/raw")
    p.add_argument("--runs", default="/home/lqs/robot-project/runs/bc")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int, default=0, help="0 表示全部")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_stats(files):
    actions, qpos = [], []
    for f in files:
        z = np.load(f)
        actions.append(z["action"])
        qpos.append(z["qpos"])
    a = np.concatenate(actions).astype(np.float32)
    q = np.concatenate(qpos).astype(np.float32)
    return {
        "action_mean": a.mean(0),
        "action_std": a.std(0) + 1e-6,
        "qpos_mean": q.mean(0),
        "qpos_std": q.std(0) + 1e-6,
    }


class EpisodeDataset(Dataset):
    """按 (episode, frame) 建索引。
    输入：当前帧图像 + 最近 history 帧 qpos
    目标：从下一帧开始、未来 chunk 步动作（修正了动作与观测的错位）
    """

    def __init__(self, files, stats, chunk, history):
        self.files = files
        self.stats = stats
        self.chunk = chunk
        self.history = history
        self.index = []
        self.lengths = []
        for fi, f in enumerate(files):
            with np.load(f) as z:
                T = len(z["images"])
            self.lengths.append(T)
            for t in range(T):
                self.index.append((fi, t))
        self.cache = {}

    def __len__(self):
        return len(self.index)

    def _episode(self, fi):
        if fi not in self.cache:
            z = np.load(self.files[fi])
            self.cache[fi] = {
                "images": z["images"],
                "qpos": z["qpos"].astype(np.float32),
                "action": z["action"].astype(np.float32),
            }
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[fi]

    def __getitem__(self, i):
        fi, t = self.index[i]
        ep = self._episode(fi)
        T = self.lengths[fi]

        img = ep["images"][t].astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        h = self.history
        t0 = max(0, t - h + 1)
        qs = ep["qpos"][t0 : t + 1]
        if len(qs) < h:
            qs = np.concatenate([np.repeat(qs[:1], h - len(qs), axis=0), qs], axis=0)
        qs = (qs - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        qs = torch.from_numpy(qs.astype(np.float32).reshape(-1))

        start = min(t + 1, T - 1)
        acts = ep["action"][start : start + self.chunk]
        if len(acts) < self.chunk:
            pad = np.repeat(acts[-1:], self.chunk - len(acts), axis=0)
            acts = np.concatenate([acts, pad], axis=0)
        acts = (acts - self.stats["action_mean"]) / self.stats["action_std"]
        return img, qs, torch.from_numpy(acts.astype(np.float32))


def main():
    args = parse_args()
    set_seed(args.seed)

    files = sorted(Path(args.data).glob("ep_*.npz"))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if len(files) < 2:
        raise SystemExit(f"数据不足: {args.data}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(files))
    n_val = max(1, int(len(files) * 0.2))
    val_files = [files[i] for i in order[:n_val]]
    train_files = [files[i] for i in order[n_val:]]

    stats = compute_stats(train_files)
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    np.savez(runs / "norm_stats.npz", **stats)
    print(f"train {len(train_files)} eps, val {len(val_files)} eps")

    train_ds = EpisodeDataset(train_files, stats, args.chunk, args.history)
    val_ds = EpisodeDataset(val_files, stats, args.chunk, args.history)
    print(f"train frames {len(train_ds)}, val frames {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = BCNet(chunk=args.chunk, qpos_dim=9 * args.history).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for img, qpos, act in train_loader:
            img, qpos, act = img.to(device), qpos.to(device), act.to(device)
            pred = model(img, qpos)
            loss = loss_fn(pred, act)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(img)
            count += len(img)

        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for img, qpos, act in val_loader:
                img, qpos, act = img.to(device), qpos.to(device), act.to(device)
                val_total += loss_fn(model(img, qpos), act).item() * len(img)
                val_count += len(img)

        train_loss = total / max(count, 1)
        val_loss = val_total / max(val_count, 1)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(f"epoch {epoch:03d}  train {train_loss:.5f}  val {val_loss:.5f}")

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "chunk": args.chunk,
                    "history": args.history,
                    "stats": {k: v.tolist() for k, v in stats.items()},
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                runs / "best.pt",
            )
            print("  saved best")

    (runs / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val {best:.5f} -> {runs / 'best.pt'}")


if __name__ == "__main__":
    main()
