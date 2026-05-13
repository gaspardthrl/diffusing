"""Watch the local checkpoints directory and push each new checkpoint to HF.

Companion to scripts/train_diffusion_merged.sh. lerobot-train saves checkpoints
locally every save_freq but only pushes the FINAL model to HF. This sidecar
watches outputs/<dir>/checkpoints/ for new step-NNNNNN directories and uploads
each one's pretrained_model/ contents to a branch named `step-NNNNNN` on the
target HF repo.

Run in a tmux pane alongside training:

    export HF_TOKEN=hf_...
    python scripts/push_checkpoints.py \\
        --checkpoints-dir outputs/diffusion-resnet-merged/checkpoints \\
        --repo gaspardthrl/diffusion-resnet-merged

Idempotent: re-running picks up where it left off. Skips checkpoints whose
branch already exists on the repo. Ignores the `last` symlink. Waits until
model.safetensors is fully written before uploading.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_branch, upload_folder
from huggingface_hub.utils import HfHubHTTPError


STEP_DIR_RE = re.compile(r"^\d{6,}$")
POLL_SECONDS = 30
SETTLE_SECONDS = 15  # extra wait after first sighting to ensure files are flushed


def list_completed_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Return checkpoint directories that look fully written (have pretrained_model/model.safetensors)."""
    if not checkpoints_dir.exists():
        return []
    found = []
    for d in sorted(checkpoints_dir.iterdir()):
        if not d.is_dir() or d.is_symlink():
            continue
        if not STEP_DIR_RE.match(d.name):
            continue
        pretrained = d / "pretrained_model"
        if (pretrained / "model.safetensors").exists() and (d / "training_state" / "training_step.json").exists():
            found.append(d)
    return found


def remote_branch_exists(api: HfApi, repo_id: str, branch: str) -> bool:
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    except HfHubHTTPError:
        return False
    return any(b.name == branch for b in refs.branches)


def upload_one(api: HfApi, repo_id: str, ckpt_dir: Path, token: str) -> str:
    """Upload <ckpt_dir>/pretrained_model/ contents to revision step-<NNN> on repo_id."""
    step = int(ckpt_dir.name)
    branch = f"step-{step}"
    pretrained = ckpt_dir / "pretrained_model"

    if not remote_branch_exists(api, repo_id, branch):
        create_branch(repo_id=repo_id, branch=branch, repo_type="model", token=token)
    upload_folder(
        repo_id=repo_id,
        folder_path=str(pretrained),
        path_in_repo=".",
        revision=branch,
        repo_type="model",
        token=token,
        commit_message=f"checkpoint @ step {step}",
    )
    return branch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", required=True,
                    help="Local checkpoints directory, e.g. outputs/diffusion-resnet-merged/checkpoints")
    ap.add_argument("--repo", required=True, help="Target HF model repo, e.g. gaspardthrl/diffusion-resnet-merged")
    ap.add_argument("--poll", type=int, default=POLL_SECONDS, help="Seconds between scans")
    ap.add_argument("--once", action="store_true",
                    help="Scan once and exit (no watch loop). Useful for bulk-push after training.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("error: set HF_TOKEN env var (write token for the repo's account)", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    ckpts_dir = Path(args.checkpoints_dir).resolve()

    # Sanity check the repo exists
    try:
        api.repo_info(repo_id=args.repo, repo_type="model")
    except HfHubHTTPError as e:
        print(f"!! cannot access repo {args.repo}: {e}", file=sys.stderr)
        sys.exit(2)

    seen: set[str] = set()
    first_seen_time: dict[str, float] = {}

    print(f"watching {ckpts_dir} → {args.repo}  (poll every {args.poll}s)", flush=True)
    while True:
        try:
            for ckpt in list_completed_checkpoints(ckpts_dir):
                if ckpt.name in seen:
                    continue
                # Wait a settle window so files are guaranteed flushed
                first_seen_time.setdefault(ckpt.name, time.time())
                age = time.time() - first_seen_time[ckpt.name]
                if age < SETTLE_SECONDS:
                    continue

                branch = f"step-{int(ckpt.name)}"
                if remote_branch_exists(api, args.repo, branch):
                    print(f"  skip {ckpt.name} — branch {branch} already on hub", flush=True)
                    seen.add(ckpt.name)
                    continue

                print(f"  uploading {ckpt.name} -> {args.repo} @ {branch} ...", flush=True)
                t0 = time.time()
                upload_one(api, args.repo, ckpt, token)
                print(f"    done in {time.time() - t0:.1f}s", flush=True)
                seen.add(ckpt.name)
        except Exception as e:
            # Don't die on transient errors - just log and retry on next poll
            print(f"  WARN: {type(e).__name__}: {e}", flush=True)

        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
