#!/usr/bin/env python
"""
Visualize the exact preprocessing applied during training for a given episode.

Current pipeline (matches train_dit_fm.sh):
  1. Resize 640×480 → 224×224  (full scene, no crop)
  2. Grayscale (always, num_output_channels=3 for CLIP)
  3. ColorJitter brightness  (shadow/lighting variation)
  4. ColorJitter contrast    (shadow/lighting variation)
  5. SharpnessJitter
  6. RandomAffine            (camera micro-movement)
  7. RandomErasing           (occlusion robustness)

Usage:
    python scripts/visualize_preprocessing.py              # episode 0, 8 frames
    python scripts/visualize_preprocessing.py --episode 5  # episode 5
    python scripts/visualize_preprocessing.py --episode 3 --n_frames 16
"""

import argparse
from pathlib import Path

import av
import torch
import torchvision.transforms.v2 as v2
from PIL import Image, ImageDraw

# ── Config (must match train_dit_fm.sh) ──────────────────────────────────────
RESIZE = (224, 224)   # --policy.image_resize_shape  (no crop after this)

# Individual augmentations shown in isolation (grayscale always applied first)
AUGMENTATIONS = {
    "brightness": v2.ColorJitter(brightness=(0.7, 1.3)),
    "contrast":   v2.ColorJitter(contrast=(0.7, 1.3)),
    "sharpness":  v2.RandomAdjustSharpness(sharpness_factor=1.4, p=1.0),
    "affine":     v2.RandomAffine(degrees=(-8, 8), translate=(0.12, 0.12)),
    "erasing":    v2.RandomErasing(p=1.0, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
}

TO_TENSOR = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
TO_PIL    = v2.ToPILImage()
GRAYSCALE = v2.Grayscale(num_output_channels=3)


def load_episode_frames(episode_idx: int, n_frames: int, data_root: str = "data") -> list:
    """Sample n_frames evenly from an episode video."""
    video_path = Path(data_root) / f"videos/observation.images.front/chunk-000/file-{episode_idx:03d}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames or 300

    indices = set(int(i * total / n_frames) for i in range(n_frames))
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            frames.append((i, frame.to_image()))
        if len(frames) == n_frames:
            break
    container.close()
    return frames


def to_base(img: Image.Image) -> torch.Tensor:
    """Resize full scene to 224×224, convert to tensor."""
    t = TO_TENSOR(img)
    return v2.Resize(RESIZE)(t)


def apply_aug(t: torch.Tensor, aug_name: str | None = None) -> Image.Image:
    """Apply grayscale (always) then optional extra augmentation, return PIL."""
    t = GRAYSCALE(t)
    if aug_name and aug_name in AUGMENTATIONS:
        t = AUGMENTATIONS[aug_name](t)
    return TO_PIL(t)


def make_frame_strip(frame_idx: int, raw: Image.Image) -> Image.Image:
    """
    Produce a horizontal strip for one frame:
    RAW | resize 224 | grayscale | +brightness | +contrast | +affine | +erasing | ALL combined
    """
    W, H = RESIZE   # 224×224 panels
    LABEL_H = 14
    PAD = 4

    base = to_base(raw)  # resized tensor, still colour

    steps = [
        ("RAW\n(original)",             TO_PIL(v2.Resize(RESIZE)(TO_TENSOR(raw)))),
        ("① MODEL\nresize 224",         TO_PIL(base)),
        ("② MODEL\ngrayscale\n(train+infer)", apply_aug(base)),
        ("③ DATASET\n+brightness\n(train only)", apply_aug(base, "brightness")),
        ("③ DATASET\n+contrast\n(train only)",   apply_aug(base, "contrast")),
        ("③ DATASET\n+affine\n(train only)",     apply_aug(base, "affine")),
        ("③ DATASET\n+erasing\n(train only)",    apply_aug(base, "erasing")),
    ]

    # ALL combined — exact training pipeline (model steps + all dataset augmentations)
    t = base.clone()
    t = GRAYSCALE(t)                                                              # model step
    t = v2.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3))(t)           # dataset aug
    t = v2.RandomAdjustSharpness(sharpness_factor=1.4, p=1.0)(t)                # dataset aug
    t = v2.RandomAffine(degrees=(-8, 8), translate=(0.12, 0.12))(t)             # dataset aug
    t = v2.RandomErasing(p=0.35, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0)(t)  # dataset aug
    steps.append(("ALL combined\n(training)", TO_PIL(t)))

    n = len(steps)
    strip_w = n * (W + PAD) + PAD
    strip_h = H + LABEL_H * 3 + PAD * 3
    strip = Image.new("RGB", (strip_w, strip_h), (30, 30, 30))
    draw = ImageDraw.Draw(strip)

    for col, (label, img_panel) in enumerate(steps):
        x = PAD + col * (W + PAD)
        y = LABEL_H * 2 + PAD
        strip.paste(img_panel, (x, y))
        for line_i, line in enumerate(label.split("\n")):
            draw.text((x + 2, 2 + line_i * 12), line, fill=(220, 220, 220))

    draw.text((2, strip_h - 14), f"frame {frame_idx}", fill=(180, 180, 180))
    return strip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode",  type=int, default=0, help="Episode index")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames to show")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--out_dir",   type=str, default="outputs/test-preprocessing")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading episode {args.episode} ({args.n_frames} frames)...")
    frames = load_episode_frames(args.episode, args.n_frames, args.data_root)

    strips = []
    for frame_idx, raw in frames:
        print(f"  Processing frame {frame_idx}...")
        strips.append(make_frame_strip(frame_idx, raw))

    total_w = strips[0].width
    total_h = sum(s.height for s in strips) + (len(strips) - 1) * 6
    canvas = Image.new("RGB", (total_w, total_h), (15, 15, 15))
    y = 0
    for strip in strips:
        canvas.paste(strip, (0, y))
        y += strip.height + 6

    out_path = out / f"episode_{args.episode:03d}_preprocessing.png"
    canvas.save(out_path)
    print(f"\nSaved → {out_path}")
    print("Columns: RAW | resize 224 | grayscale | +brightness | +contrast | +affine | +erasing | ALL combined")


if __name__ == "__main__":
    main()
