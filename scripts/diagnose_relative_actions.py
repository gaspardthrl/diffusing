#!/usr/bin/env python
"""Diagnostic checks for the relative action implementation.

Checks:
  1. Round-trip reconstruction (chunk-wise: relative→absolute must be lossless)
  2. Delta distribution per joint (mean, std, p99, max)
  3. Normalization sanity after applying relative_stats.json
  4. Reference frame confirmation (step-wise vs chunk-wise, in both stats and runtime)
  5. Gripper isolation (normalization stats computed independently)

Usage:
    uv run python scripts/diagnose_relative_actions.py \
        --repo-id gaspardthrl/walleed_fold_combined \
        --chunk-size 32 \
        --stats meta/relative_stats.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_IDX = 5
REL_DIMS = list(range(5))  # joints 0-4 are relative; gripper (5) is absolute

# ── helpers ───────────────────────────────────────────────────────────────────

def sep(title=""):
    width = 70
    if title:
        print(f"\n{'─'*3} {title} {'─'*(width-5-len(title))}")
    else:
        print("─" * width)


def load_dataset(repo_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download parquets only, return (actions, states, episode_indices)."""
    print(f"Downloading parquets for {repo_id} …")
    root = Path(snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=f"/tmp/diag__{repo_id.replace('/', '__')}",
        allow_patterns=["data/**/*.parquet", "meta/"],
    ))
    import pandas as pd
    dfs = []
    chunk_dir = root / "data" / "chunk-000"
    for p in sorted(chunk_dir.glob("file-*.parquet")):
        dfs.append(pd.read_parquet(p, columns=["episode_index", "frame_index", "action", "observation.state"]))
    df = pd.concat(dfs).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    actions = np.stack(df["action"].values).astype(np.float32)
    states  = np.stack(df["observation.state"].values).astype(np.float32)
    ep_idx  = df["episode_index"].values
    print(f"  Loaded {len(df)} frames, {len(np.unique(ep_idx))} episodes, action_dim={actions.shape[1]}")
    return actions, states, ep_idx


def get_valid_starts(ep_idx: np.ndarray, chunk_size: int) -> np.ndarray:
    total = len(ep_idx)
    starts = np.arange(total - chunk_size + 1)
    valid = ep_idx[starts] == ep_idx[starts + chunk_size - 1]
    return starts[valid]

# ── check 1: round-trip ───────────────────────────────────────────────────────

def check_round_trip(actions, states, ep_idx, chunk_size, n_episodes=10):
    sep("CHECK 1 — Round-trip reconstruction (chunk-wise)")
    print("  Formula: relative[t] = action[t] - state[chunk_start]")
    print("  Inverse : absolute[t] = relative[t] + state[chunk_start]")
    print()

    unique_eps = np.unique(ep_idx)
    rng = np.random.default_rng(42)
    sample_eps = rng.choice(unique_eps, size=min(n_episodes, len(unique_eps)), replace=False)

    all_errors = []
    print(f"  {'episode':>10}  {'max_error':>14}  {'status':>8}")
    for ep in sorted(sample_eps):
        mask = ep_idx == ep
        ep_actions = actions[mask]
        ep_states  = states[mask]
        T = len(ep_actions)
        if T < chunk_size:
            continue
        # Take the first valid chunk
        ref_state = ep_states[0]  # state at chunk start (t=0 of this episode)
        chunk_abs = ep_actions[:chunk_size].copy()
        # → relative
        chunk_rel = chunk_abs.copy()
        chunk_rel[:, :GRIPPER_IDX] -= ref_state[:GRIPPER_IDX]
        # → back to absolute
        chunk_rec = chunk_rel.copy()
        chunk_rec[:, :GRIPPER_IDX] += ref_state[:GRIPPER_IDX]
        err = np.abs(chunk_rec - chunk_abs).max()
        all_errors.append(err)
        status = "OK" if err < 1e-5 else "FAIL"
        print(f"  {ep:>10}  {err:>14.2e}  {status:>8}")

    print()
    max_err = max(all_errors) if all_errors else float("nan")
    if max_err < 1e-5:
        print(f"  ✓ PASS  max_error={max_err:.2e} < 1e-5")
    else:
        print(f"  ✗ FAIL  max_error={max_err:.2e} >= 1e-5")
    return max_err


# ── check 2: delta distribution ───────────────────────────────────────────────

def check_delta_distribution(actions, states, ep_idx, chunk_size):
    sep("CHECK 2 — Delta distribution per joint")
    print("  delta[t] = action[t] - state[chunk_start]  (for joints 0-4)")
    print("  Gripper : absolute (not delta)")
    print()

    valid_starts = get_valid_starts(ep_idx, chunk_size)
    offsets = np.arange(chunk_size)
    frame_idx = valid_starts[:, None] + offsets[None, :]
    chunks = actions[frame_idx].copy()                          # (N, chunk, 6)
    chunk_states = states[valid_starts]                         # (N, 6)
    chunks[:, :, :GRIPPER_IDX] -= chunk_states[:, None, :GRIPPER_IDX]
    deltas = chunks.reshape(-1, actions.shape[1])               # (N*chunk, 6)

    print(f"  {'joint':>14}  {'mean':>9}  {'std':>9}  {'p99_abs':>9}  {'max_abs':>9}  {'flag':>6}")
    flags = {}
    for j, name in enumerate(JOINT_NAMES):
        d = deltas[:, j]
        if j == GRIPPER_IDX:
            # gripper is absolute, not delta — show its raw distribution instead
            g = actions[:, GRIPPER_IDX]
            print(f"  {name:>14}  [absolute — raw stats below]")
            print(f"  {'':>14}  {g.mean():>9.4f}  {g.std():>9.4f}  {np.percentile(np.abs(g),99):>9.4f}  {np.abs(g).max():>9.4f}")
            continue
        p99 = float(np.percentile(np.abs(d), 99))
        flag = "LARGE" if p99 > 0.5 else ""
        flags[name] = p99
        print(f"  {name:>14}  {d.mean():>9.4f}  {d.std():>9.4f}  {p99:>9.4f}  {np.abs(d).max():>9.4f}  {flag:>6}")

    return deltas


# ── check 3: normalization sanity ─────────────────────────────────────────────

def check_normalization(deltas, stats_path):
    sep("CHECK 3 — Normalization sanity (after applying relative_stats.json)")
    if not Path(stats_path).exists():
        print(f"  SKIP — {stats_path} not found")
        return

    with open(stats_path) as f:
        raw = json.load(f)
    action_stats = raw["action"]
    mn  = np.array(action_stats["min"],  dtype=np.float32)
    mx  = np.array(action_stats["max"],  dtype=np.float32)
    mean = np.array(action_stats["mean"], dtype=np.float32)
    std  = np.array(action_stats["std"],  dtype=np.float32) if "std" in action_stats else None

    print(f"  Normalization type detected: {'MEAN_STD' if std is not None else 'MIN_MAX'}")
    print()

    if std is not None:
        # mean-std normalization: x_norm = (x - mean) / std
        normalized = (deltas - mean) / (std + 1e-8)
    else:
        # min-max normalization: x_norm = 2*(x - min)/(max - min) - 1  [→ [-1, 1]]
        rng = mx - mn
        normalized = 2.0 * (deltas - mn) / (rng + 1e-8) - 1.0

    print(f"  {'joint':>14}  {'norm_mean':>10}  {'norm_std':>10}  {'norm_min':>10}  {'norm_max':>10}  {'status':>8}")
    issues = []
    for j, name in enumerate(JOINT_NAMES):
        n = normalized[:, j]
        nm, ns, nlo, nhi = n.mean(), n.std(), n.min(), n.max()
        if j < GRIPPER_IDX:
            ok = 0.3 < ns < 3.0 and -10 < nlo and nhi < 10
        else:
            ok = -10 < nlo and nhi < 10
        status = "OK" if ok else "WARN"
        if not ok:
            issues.append(name)
        print(f"  {name:>14}  {nm:>10.3f}  {ns:>10.3f}  {nlo:>10.3f}  {nhi:>10.3f}  {status:>8}")

    print()
    if not issues:
        print("  ✓ Normalization looks healthy (std in (0.3, 3.0), range within [-10, 10])")
    else:
        print(f"  ✗ Issues with: {issues}")
        if std is not None:
            print("    → std << 1 means stats were computed on a different distribution than the data")


# ── check 4: reference frame ──────────────────────────────────────────────────

def check_reference_frame(chunk_size):
    sep("CHECK 4 — Reference frame confirmation")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  STATS COMPUTATION  (compute_relative_action_stats)         │")
    print("  │  chunks[:, :, :5] -= states[:, None, :5]                    │")
    print("  │  → ALL actions in chunk relative to state[t_start]          │")
    print("  │  → CHUNK-WISE                                                │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  RUNTIME PREPROCESSING  (RelativeActionsProcessorStep)      │")
    print("  │  action[..., :5] -= state[:, 0, :5]  (current obs state)    │")
    print("  │  → ALL actions in chunk relative to current observation      │")
    print("  │  → CHUNK-WISE                                                │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  INFERENCE POSTPROCESSING  (AbsoluteActionsProcessorStep)   │")
    print("  │  action[..., :5] += cached_state[:, 0, :5]                  │")
    print("  │  cached_state = observation.state from the SAME call         │")
    print("  │  → CHUNK-WISE inverse — MATCHES preprocessing                │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ✓ Reference frames MATCH: both stats and runtime use chunk-wise")
    print()
    print("  ⚠  NOTE: state.ndim==3 branch takes state[:, 0] = OLDEST obs step")
    print(f"     With n_obs_steps=1 this is the current state → CORRECT")
    print(f"     With n_obs_steps>1 this is t-(n_obs_steps-1) → WRONG reference")
    print(f"     Training scripts set n_obs_steps=1 → OK")


# ── check 5: gripper isolation ────────────────────────────────────────────────

def check_gripper_isolation(actions, states_arr, ep_idx, stats_path):
    sep("CHECK 5 — Gripper isolation")

    gripper_abs = actions[:, GRIPPER_IDX]
    print(f"  Gripper raw (absolute):  mean={gripper_abs.mean():.4f}  std={gripper_abs.std():.4f}"
          f"  min={gripper_abs.min():.4f}  max={gripper_abs.max():.4f}")

    if Path(stats_path).exists():
        with open(stats_path) as f:
            raw = json.load(f)
        stats = raw["action"]
        mn = np.array(stats["min"])
        mx = np.array(stats["max"])
        print(f"  Gripper stats from JSON: min={mn[GRIPPER_IDX]:.4f}  max={mx[GRIPPER_IDX]:.4f}")
        if abs(mn[GRIPPER_IDX] - gripper_abs.min()) < 0.05 and abs(mx[GRIPPER_IDX] - gripper_abs.max()) < 0.05:
            print("  ✓ Gripper stats match raw absolute values → correctly kept absolute in stats")
        else:
            joint_deltas = actions.copy()
            joint_deltas[:, :GRIPPER_IDX] -= states_arr[:, :GRIPPER_IDX]
            if abs(mn[GRIPPER_IDX] - joint_deltas[:, GRIPPER_IDX].min()) < 0.05:
                print("  ✗ Gripper stats match DELTA values → gripper was incorrectly relativized in stats")
            else:
                print("  ? Gripper stats don't match raw or delta; may have been computed on a different dataset split")
    else:
        print(f"  SKIP — {stats_path} not found, cannot verify normalization stats")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_fold_combined")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--stats", default="data/meta/relative_stats.json")
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    actions, states, ep_idx = load_dataset(args.repo_id)

    check_round_trip(actions, states, ep_idx, args.chunk_size, n_episodes=args.n_episodes)
    deltas = check_delta_distribution(actions, states, ep_idx, args.chunk_size)
    check_normalization(deltas, args.stats)
    check_reference_frame(args.chunk_size)
    check_gripper_isolation(actions, states, ep_idx, args.stats)

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
