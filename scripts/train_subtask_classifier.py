#!/usr/bin/env python3
"""
Subtask classifier: frozen backbone + state MLP + prev-subtask embedding → 8 classes.
Trains on VLM annotations from annotate_sarm.sh.

Backbones: clip | dinov2_s | dinov2_b | resnet18 | resnet50

Usage:
    python scripts/train_subtask_classifier.py \
        --dataset-root ~/.cache/huggingface/lerobot/hub/datasets--gaspardthrl--walleed_hg_double_fold_clean/snapshots/<hash> \
        --backbone clip \
        --output-dir outputs/cls_clip
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SUBTASK_NAMES = [
    "reach the towel first fold",
    "pick the towel first fold",
    "perform first fold",
    "release towel first fold",
    "reach the towel second fold",
    "pick the towel second fold",
    "perform second fold",
    "release towel second fold",
]
N_CLASSES = len(SUBTASK_NAMES)
STATE_DIM = 6
IMAGE_KEY = "observation.images.front"
FPS = 30

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Per-worker video capture cache — each DataLoader worker process has its own copy
_caps: dict[str, cv2.VideoCapture] = {}


def _read_frame(video_path: str, abs_frame: int) -> np.ndarray:
    if video_path not in _caps or not _caps[video_path].isOpened():
        _caps[video_path] = cv2.VideoCapture(video_path)
    cap = _caps[video_path]
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {abs_frame} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def apply_grayworld(x: torch.Tensor) -> torch.Tensor:
    """x: (C, H, W) float [0, 1]"""
    means = x.mean(dim=(-2, -1), keepdim=True)
    overall = means.mean(dim=0, keepdim=True)
    return (x * (overall / (means + 1e-6))).clamp(0.0, 1.0)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_annotations(dataset_root: Path) -> dict[int, np.ndarray]:
    """Returns {ep_idx: per-frame int8 stage label array}."""
    out = {}
    for f in sorted(dataset_root.glob("meta/episodes/**/*.parquet")):
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            names = row.get("dense_subtask_names")
            if not isinstance(names, (list, np.ndarray)):
                continue
            ep = int(row["episode_index"])
            starts = [int(x) for x in row["dense_subtask_start_frames"]]
            ends = [int(x) for x in row["dense_subtask_end_frames"]]
            n = max(ends[-1] + 1, int(row.get("length") or 0))
            labels = np.zeros(n, dtype=np.int8)
            for name, s, e in zip(names, starts, ends):
                try:
                    labels[s : e + 1] = SUBTASK_NAMES.index(name)
                except ValueError:
                    pass
            out[ep] = labels
    return out


def load_episode_meta(dataset_root: Path) -> dict[int, dict]:
    """Returns {ep_idx: {chunk_idx, file_idx, from_ts, length}}."""
    out = {}
    ck = f"videos/{IMAGE_KEY}/chunk_index"
    fk = f"videos/{IMAGE_KEY}/file_index"
    tk = f"videos/{IMAGE_KEY}/from_timestamp"
    for f in sorted(dataset_root.glob("meta/episodes/**/*.parquet")):
        df = pd.read_parquet(f, columns=["episode_index", "length", ck, fk, tk])
        for _, row in df.iterrows():
            out[int(row["episode_index"])] = {
                "chunk_idx": int(row[ck]),
                "file_idx": int(row[fk]),
                "from_ts": float(row[tk]),
                "length": int(row["length"]),
            }
    return out


def load_all_states(dataset_root: Path) -> np.ndarray:
    """Returns states array of shape (total_frames, STATE_DIM) indexed by global frame index."""
    log.info("Loading states from data parquets...")
    # First pass: find max global index
    max_idx = 0
    for f in sorted(dataset_root.glob("data/**/*.parquet")):
        df = pd.read_parquet(f, columns=["index"])
        max_idx = max(max_idx, int(df["index"].max()))

    states = np.zeros((max_idx + 1, STATE_DIM), dtype=np.float32)
    for f in sorted(dataset_root.glob("data/**/*.parquet")):
        df = pd.read_parquet(f, columns=["index", "observation.state"])
        indices = df["index"].values.astype(int)
        vals = np.stack(df["observation.state"].values).astype(np.float32)
        states[indices] = vals
    log.info(f"  {max_idx + 1} frames loaded")
    return states


# ── Dataset ────────────────────────────────────────────────────────────────────

class SubtaskDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        annotations: dict[int, np.ndarray],
        ep_meta: dict[int, dict],
        states: np.ndarray,          # (N_total, 6)
        ep_from_index: dict[int, int],  # ep_idx -> global start index
        backbone: str,
        split_eps: set[int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        subsample: int = 3,
    ):
        self.dataset_root = dataset_root
        self.annotations = annotations
        self.ep_meta = ep_meta
        self.states = states
        self.ep_from_index = ep_from_index
        self.backbone = backbone
        self.state_mean = state_mean
        self.state_std = state_std

        # (ep_idx, frame_in_ep, stage, prev_stage)
        self.samples: list[tuple[int, int, int, int]] = []
        for ep in sorted(split_eps):
            labels = annotations[ep]
            for t in range(0, len(labels), subsample):
                prev = int(labels[t - 1]) if t > 0 else int(labels[0])
                self.samples.append((ep, t, int(labels[t]), prev))

        self.resize = transforms.Resize((224, 224), antialias=True)
        mean, std = (CLIP_MEAN, CLIP_STD) if backbone == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ep, t, stage, prev_stage = self.samples[idx]
        vm = self.ep_meta[ep]

        # Image: seek to abs frame in concatenated video file
        vpath = str(
            self.dataset_root
            / f"videos/{IMAGE_KEY}/chunk-{vm['chunk_idx']:03d}/file-{vm['file_idx']:03d}.mp4"
        )
        abs_frame = round(vm["from_ts"] * FPS) + t
        frame = _read_frame(vpath, abs_frame)
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img = self.resize(img)
        img = apply_grayworld(img)
        img = self.normalize(img)

        # State: normalise
        global_idx = self.ep_from_index[ep] + t
        raw = self.states[global_idx] if global_idx < len(self.states) else np.zeros(STATE_DIM, np.float32)
        state = torch.from_numpy((raw - self.state_mean) / (self.state_std + 1e-6))

        return img, state, torch.tensor(prev_stage), torch.tensor(stage)

    @property
    def class_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for _, _, stage, _ in self.samples:
            counts[stage] += 1
        return counts


# ── Model ──────────────────────────────────────────────────────────────────────

class FrozenBackbone(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "clip":
            from transformers import CLIPVisionModel
            self._m = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.feat_dim = 512
        elif name == "dinov2_s":
            self._m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.feat_dim = 384
        elif name == "dinov2_b":
            self._m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.feat_dim = 768
        elif name == "resnet18":
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            self._m = nn.Sequential(*list(m.children())[:-1])
            self.feat_dim = 512
        elif name == "resnet50":
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            self._m = nn.Sequential(*list(m.children())[:-1])
            self.feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {name}")
        for p in self._m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "clip":
            return self._m(pixel_values=x).pooler_output
        elif self.name.startswith("dinov2"):
            return self._m(x)
        else:
            return self._m(x).flatten(1)


class SubtaskClassifier(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = FrozenBackbone(backbone)
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_DIM, 64), nn.ReLU(), nn.Linear(64, 64),
        )
        self.subtask_emb = nn.Embedding(N_CLASSES, 16)
        head_in = self.backbone.feat_dim + 64 + 16
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, N_CLASSES),
        )

    def forward(self, img: torch.Tensor, state: torch.Tensor, prev_subtask: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([
            self.backbone(img),
            self.state_enc(state),
            self.subtask_emb(prev_subtask),
        ], dim=-1))


# ── Training ───────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)
    root = Path(args.dataset_root)

    annotations = load_annotations(root)
    if not annotations:
        raise RuntimeError("No dense annotations found. Run annotate_sarm.sh first.")
    log.info(f"Annotations: {len(annotations)} episodes")

    ep_meta = load_episode_meta(root)
    states = load_all_states(root)

    # ep_from_index from episode meta
    ep_from_index: dict[int, int] = {}
    for f in sorted(root.glob("meta/episodes/**/*.parquet")):
        df = pd.read_parquet(f, columns=["episode_index", "dataset_from_index"])
        for _, row in df.iterrows():
            ep_from_index[int(row["episode_index"])] = int(row["dataset_from_index"])

    # Episode split
    all_eps = sorted(set(annotations) & set(ep_meta))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(all_eps))
    n_val = max(1, int(len(all_eps) * 0.15))
    val_eps = {all_eps[i] for i in perm[:n_val]}
    train_eps = {all_eps[i] for i in perm[n_val:]}
    log.info(f"Split — train: {len(train_eps)} eps, val: {len(val_eps)} eps")

    # State normalisation from training data
    train_indices = [
        ep_from_index[ep] + t
        for ep in train_eps
        for t in range(len(annotations[ep]))
        if ep_from_index.get(ep, -1) + t < len(states)
    ]
    train_state_arr = states[train_indices]
    state_mean = train_state_arr.mean(axis=0)
    state_std = train_state_arr.std(axis=0)

    common_kwargs = dict(
        dataset_root=root, annotations=annotations, ep_meta=ep_meta,
        states=states, ep_from_index=ep_from_index, backbone=args.backbone,
        state_mean=state_mean, state_std=state_std, subsample=args.subsample,
    )
    train_ds = SubtaskDataset(split_eps=train_eps, **common_kwargs)
    val_ds = SubtaskDataset(split_eps=val_eps, **common_kwargs)
    log.info(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}")

    # Weighted sampler for class balance
    counts = train_ds.class_counts
    log.info("Train class counts: " + ", ".join(f"{SUBTASK_NAMES[i][:20]}: {counts[i]}" for i in range(N_CLASSES)))
    w = 1.0 / np.maximum(counts, 1).astype(float)
    sample_w = torch.tensor([w[s] for _, _, s, _ in train_ds.samples], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_w, len(sample_w))

    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    model = SubtaskClassifier(args.backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tloss, tcorr, tn = 0.0, 0, 0
        for img, state, prev_st, label in train_loader:
            img, state, prev_st, label = img.to(device), state.to(device), prev_st.to(device), label.to(device)
            logits = model(img, state, prev_st)
            loss = nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item() * len(label)
            tcorr += (logits.argmax(1) == label).sum().item()
            tn += len(label)
        scheduler.step()

        model.eval()
        vcorr, vn = 0, 0
        pc_correct = np.zeros(N_CLASSES)
        pc_total = np.zeros(N_CLASSES)
        with torch.no_grad():
            for img, state, prev_st, label in val_loader:
                img, state, prev_st, label = img.to(device), state.to(device), prev_st.to(device), label.to(device)
                preds = model(img, state, prev_st).argmax(1)
                vcorr += (preds == label).sum().item()
                vn += len(label)
                for c in range(N_CLASSES):
                    mask = label == c
                    pc_correct[c] += (preds[mask] == c).sum().item()
                    pc_total[c] += mask.sum().item()

        val_acc = vcorr / vn
        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {tloss/tn:.4f} | train {tcorr/tn:.3f} | val {val_acc:.3f}"
        )
        if epoch % 5 == 0 or epoch == args.epochs:
            pc_acc = pc_correct / np.maximum(pc_total, 1)
            for i, name in enumerate(SUBTASK_NAMES):
                log.info(f"  [{i}] {name[:38]:38s}  {pc_acc[i]:.3f}  (n={int(pc_total[i])})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "backbone": args.backbone, "val_acc": val_acc,
                "state_mean": state_mean, "state_std": state_std,
            }, out / "best.pt")

    torch.save({
        "epoch": args.epochs, "model": model.state_dict(),
        "backbone": args.backbone, "val_acc": val_acc,
        "state_mean": state_mean, "state_std": state_std,
    }, out / "final.pt")
    json.dump(
        {"backbone": args.backbone, "best_val_acc": best_val_acc, "subtask_names": SUBTASK_NAMES},
        open(out / "config.json", "w"), indent=2,
    )
    log.info(f"Best val acc: {best_val_acc:.3f}  →  {out}/best.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, help="Path to annotated dataset snapshot")
    p.add_argument("--backbone", default="clip",
                   choices=["clip", "dinov2_s", "dinov2_b", "resnet18", "resnet50"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subsample", type=int, default=3,
                   help="Use every Nth frame — 3 means effective 10fps (default: 3)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__ == "__main__":
    main()
