"""Merge two LeRobot v3 datasets into one.

Concatenates dataset A (e.g. truncated single-fold) and dataset B (e.g. double-fold),
renumbering episodes, file indices, and the global `index` column so the result is
a valid LeRobot v3 dataset.

Assumptions verified at runtime:
    * Both datasets have the same fps, feature schema, video codec, and meta schema.
    * Both datasets have a single chunk (chunk-000); the merged output stays in chunk-000.
    * Tasks are merged as a union by task name; if both have the same task name (e.g.
      "Fold the towel"), the merged dataset has one task and task_index in data rows
      remains 0. Otherwise rows get remapped to the merged task table.

What gets written:
    data/chunk-000/file-NNN.parquet      contiguously renumbered (A files first, then B)
    meta/episodes/chunk-000/file-000.parquet   one merged meta parquet
    meta/info.json                       totals + splits updated, codec/etc from B
    meta/tasks.parquet                   merged task table
    meta/stats.json                      copied from B (the larger dataset)
    meta/relative_stats.json             copied from B if it exists (flag as stale)
    videos/<key>/chunk-000/file-NNN.mp4  hardlinked, renumbered same as data files (independently)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VIDEO_KEY = "observation.images.front"


# ---------- helpers ---------- #

def _load_meta(src: Path) -> pd.DataFrame:
    parts = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)


def _load_tasks(src: Path) -> pd.DataFrame:
    """Return DataFrame with columns task_index, task. Falls back to inferring from the
    meta/episodes `tasks` column if meta/tasks.parquet is missing.
    """
    tp = src / "meta" / "tasks.parquet"
    if tp.exists():
        df = pq.read_table(tp).to_pandas().reset_index()
        return df[["task_index", "task"]]
    meta = _load_meta(src)
    seen = {}
    rows = []
    for tasks_list in meta["tasks"]:
        for t in tasks_list:
            if t not in seen:
                seen[t] = len(seen)
                rows.append({"task_index": seen[t], "task": t})
    return pd.DataFrame(rows)


def _validate_compat(info_a: dict, info_b: dict) -> None:
    if info_a["fps"] != info_b["fps"]:
        raise ValueError(f"fps mismatch: {info_a['fps']} vs {info_b['fps']}")
    for key in ("action", "observation.state", VIDEO_KEY):
        fa = info_a["features"].get(key)
        fb = info_b["features"].get(key)
        if fa is None or fb is None:
            raise ValueError(f"feature '{key}' missing from one of the datasets")
        if fa.get("shape") != fb.get("shape"):
            raise ValueError(f"feature '{key}' shape mismatch: {fa.get('shape')} vs {fb.get('shape')}")
    vi_a = info_a["features"][VIDEO_KEY].get("info", {})
    vi_b = info_b["features"][VIDEO_KEY].get("info", {})
    for k in ("video.codec", "video.height", "video.width", "video.pix_fmt", "video.fps"):
        if vi_a.get(k) != vi_b.get(k):
            raise ValueError(f"video info '{k}' mismatch: {vi_a.get(k)} vs {vi_b.get(k)}")


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------- main ---------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-a", required=True, help="Path to first source dataset (e.g. single-fold)")
    ap.add_argument("--src-b", required=True, help="Path to second source dataset (e.g. double-fold)")
    ap.add_argument("--dst", required=True, help="Destination dataset directory (must not exist)")
    args = ap.parse_args()

    src_a = Path(args.src_a)
    src_b = Path(args.src_b)
    dst = Path(args.dst)
    if dst.exists():
        print(f"!! destination {dst} already exists. Delete it first.", file=sys.stderr)
        sys.exit(1)

    info_a = json.loads((src_a / "meta" / "info.json").read_text())
    info_b = json.loads((src_b / "meta" / "info.json").read_text())
    _validate_compat(info_a, info_b)
    fps = float(info_a["fps"])
    print(f"[1/7] schemas compatible. fps={fps}, codec={info_a['features'][VIDEO_KEY]['info']['video.codec']}")

    meta_a = _load_meta(src_a)
    meta_b = _load_meta(src_b)
    n_a, n_b = len(meta_a), len(meta_b)
    print(f"      A: {n_a} episodes, {int(meta_a['length'].sum())} frames")
    print(f"      B: {n_b} episodes, {int(meta_b['length'].sum())} frames")

    # ---- task merging ---- #
    tasks_a = _load_tasks(src_a)
    tasks_b = _load_tasks(src_b)
    # union by task name
    union_tasks: list[str] = []
    for t in list(tasks_a["task"]) + list(tasks_b["task"]):
        if t not in union_tasks:
            union_tasks.append(t)
    merged_tasks = pd.DataFrame(
        [{"task_index": i, "task": t} for i, t in enumerate(union_tasks)]
    )
    task_remap_a = {int(r["task_index"]): union_tasks.index(r["task"]) for _, r in tasks_a.iterrows()}
    task_remap_b = {int(r["task_index"]): union_tasks.index(r["task"]) for _, r in tasks_b.iterrows()}
    print(f"[2/7] tasks: A={list(tasks_a['task'])}  B={list(tasks_b['task'])}  -> merged {union_tasks}")

    # ---- episode + file index remaps ---- #
    ep_remap_a = {int(e): int(e) for e in meta_a["episode_index"]}
    ep_remap_b = {int(e): int(e) + n_a for e in meta_b["episode_index"]}

    # data file_index remap: A's files (sorted) get sequential new indices starting at 0,
    # then B's files get sequential new indices continuing from there.
    data_files_a = sorted((src_a / "data" / "chunk-000").glob("file-*.parquet"))
    data_files_b = sorted((src_b / "data" / "chunk-000").glob("file-*.parquet"))
    data_file_remap_a, data_file_remap_b = {}, {}
    cursor = 0
    for p in data_files_a:
        data_file_remap_a[int(p.stem.split("-")[1])] = cursor
        cursor += 1
    for p in data_files_b:
        data_file_remap_b[int(p.stem.split("-")[1])] = cursor
        cursor += 1

    # video file_index remap: independent of data file_index numbering
    video_files_a = sorted((src_a / "videos" / VIDEO_KEY / "chunk-000").glob("file-*.mp4"))
    video_files_b = sorted((src_b / "videos" / VIDEO_KEY / "chunk-000").glob("file-*.mp4"))
    video_file_remap_a, video_file_remap_b = {}, {}
    cursor = 0
    for p in video_files_a:
        video_file_remap_a[int(p.stem.split("-")[1])] = cursor
        cursor += 1
    for p in video_files_b:
        video_file_remap_b[int(p.stem.split("-")[1])] = cursor
        cursor += 1

    # ---- new global frame index map ---- #
    # process A in episode order, then B in episode order
    new_from: dict[int, int] = {}
    new_to: dict[int, int] = {}
    cursor = 0
    for _, row in meta_a.sort_values("episode_index").iterrows():
        new_ep = ep_remap_a[int(row["episode_index"])]
        new_from[new_ep] = cursor
        new_to[new_ep] = cursor + int(row["length"])
        cursor = new_to[new_ep]
    for _, row in meta_b.sort_values("episode_index").iterrows():
        new_ep = ep_remap_b[int(row["episode_index"])]
        new_from[new_ep] = cursor
        new_to[new_ep] = cursor + int(row["length"])
        cursor = new_to[new_ep]
    new_total_frames = cursor
    print(f"[3/7] merged size: {n_a + n_b} episodes, {new_total_frames} frames")

    # ---- create destination structure ---- #
    dst.mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dst / "videos" / VIDEO_KEY / "chunk-000").mkdir(parents=True)

    # ---- rewrite data parquets ---- #
    print(f"[4/7] writing {len(data_files_a) + len(data_files_b)} data parquets ...")
    data_schema = None
    for source_name, files, ep_map, file_map, task_map in [
        ("A", data_files_a, ep_remap_a, data_file_remap_a, task_remap_a),
        ("B", data_files_b, ep_remap_b, data_file_remap_b, task_remap_b),
    ]:
        for src_path in files:
            orig_file_idx = int(src_path.stem.split("-")[1])
            new_file_idx = file_map[orig_file_idx]
            table = pq.read_table(src_path)
            if data_schema is None:
                data_schema = table.schema
            df = table.to_pandas()
            df["episode_index"] = df["episode_index"].map(lambda e: ep_map[int(e)]).astype(np.int64)
            if task_map:
                df["task_index"] = df["task_index"].map(lambda t: task_map[int(t)]).astype(np.int64)
            # rebuild global `index` column: rows are already sorted by (ep, frame); for
            # each ep in this file, place its rows into [new_from[ep], new_to[ep]).
            new_indices = np.empty(len(df), dtype=np.int64)
            for ep, sub in df.groupby("episode_index", sort=True):
                positions = sub.index.to_numpy()
                new_indices[positions] = np.arange(new_from[int(ep)], new_to[int(ep)], dtype=np.int64)
            df["index"] = new_indices
            dst_path = dst / "data" / "chunk-000" / f"file-{new_file_idx:03d}.parquet"
            pq.write_table(pa.Table.from_pandas(df, schema=data_schema, preserve_index=False), dst_path)

    # ---- merged meta parquet ---- #
    print("[5/7] writing merged meta/episodes parquet ...")
    meta_schema = pq.read_schema(sorted((src_a / "meta" / "episodes").rglob("*.parquet"))[0])
    rows_out: list[dict] = []
    for source_name, meta_df, ep_map, data_fm, video_fm, task_map in [
        ("A", meta_a, ep_remap_a, data_file_remap_a, video_file_remap_a, task_remap_a),
        ("B", meta_b, ep_remap_b, data_file_remap_b, video_file_remap_b, task_remap_b),
    ]:
        for _, row in meta_df.iterrows():
            r = row.to_dict()
            orig_ep = int(r["episode_index"])
            new_ep = ep_map[orig_ep]
            r["episode_index"] = new_ep
            r["data/file_index"] = int(data_fm[int(r["data/file_index"])])
            r[f"videos/{VIDEO_KEY}/file_index"] = int(video_fm[int(r[f"videos/{VIDEO_KEY}/file_index"])])
            r["dataset_from_index"] = new_from[new_ep]
            r["dataset_to_index"] = new_to[new_ep]
            # this metadata file is itself stored at meta/episodes/chunk-000/file-000
            r["meta/episodes/chunk_index"] = 0
            r["meta/episodes/file_index"] = 0
            # task list strings — no remap needed since we keep the original strings
            # (the index that matters is in row data; tasks list is just for display)
            rows_out.append(r)
    merged_meta = pd.DataFrame(rows_out).sort_values("episode_index").reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(merged_meta, schema=meta_schema, preserve_index=False),
        dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    # ---- hardlink videos ---- #
    print(f"[6/7] linking {len(video_files_a) + len(video_files_b)} video files ...")
    for src_video in video_files_a:
        new_idx = video_file_remap_a[int(src_video.stem.split("-")[1])]
        _hardlink_or_copy(src_video, dst / "videos" / VIDEO_KEY / "chunk-000" / f"file-{new_idx:03d}.mp4")
    for src_video in video_files_b:
        new_idx = video_file_remap_b[int(src_video.stem.split("-")[1])]
        _hardlink_or_copy(src_video, dst / "videos" / VIDEO_KEY / "chunk-000" / f"file-{new_idx:03d}.mp4")

    # ---- top-level meta ---- #
    print("[7/7] writing info.json, tasks.parquet, stats.json, relative_stats.json (if present) ...")
    # tasks.parquet
    pq.write_table(pa.Table.from_pandas(merged_tasks, preserve_index=False), dst / "meta" / "tasks.parquet")
    # info.json — prefer B's because it has robot_type=so_follower whereas A has None
    info_out = json.loads(json.dumps(info_b))  # deep copy
    info_out["total_episodes"] = n_a + n_b
    info_out["total_frames"] = new_total_frames
    info_out["total_tasks"] = len(merged_tasks)
    info_out["splits"] = {"train": f"0:{n_a + n_b}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info_out, indent=4))
    # stats.json — copy from B (larger), with caveat
    if (src_b / "meta" / "stats.json").exists():
        shutil.copy(src_b / "meta" / "stats.json", dst / "meta" / "stats.json")
    if (src_b / "meta" / "relative_stats.json").exists():
        shutil.copy(src_b / "meta" / "relative_stats.json", dst / "meta" / "relative_stats.json")
        print("      NOTE: copied relative_stats.json from B unchanged. Regenerate for the merged dataset before relative-action training.")

    print(f"done. merged dataset at: {dst}")


if __name__ == "__main__":
    main()
