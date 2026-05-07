# diffusing

SO-101 imitation learning with async inference support via [LeRobot](https://github.com/huggingface/lerobot).

The repo wraps [LeRobot](https://github.com/huggingface/lerobot) (vendored as a git submodule under `third_party/lerobot`) with thin shell scripts for the SO-101 leader/follower workflow: motor setup, calibration, teleoperation, dataset recording, policy training, and async inference (policy server + robot client).

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-username>/diffusing.git
cd diffusing
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 3. Install dependencies

For robot control only:

```bash
uv sync --extra robot
```

For async inference (policy server + client):

```bash
uv sync --extra robot --extra inference
```

### 4. Configure `.env`

The local scripts in `scripts/` source a `.env` file at the repo root. Create one with your serial ports and arm IDs:

```bash
# .env
FOLLOWER_PORT=/dev/tty.usbmodem<FOLLOWER_SERIAL>   # e.g. /dev/tty.usbmodem5B141136511
LEADER_PORT=/dev/tty.usbmodem<LEADER_SERIAL>       # e.g. /dev/tty.usbmodem5B141129431
FOLLOWER_ID=<your-follower-id>                     # e.g. walleed_follower
LEADER_ID=<your-leader-id>                         # e.g. walleed_leader
HF_TOKEN=<your-huggingface-token>                  # needed to push datasets / policies
```

To find the serial port for a given controller, plug it in alone and run:

```bash
ls /dev/tty.usbmodem*    # macOS
ls /dev/ttyACM*          # Linux
```

The arm IDs are arbitrary labels — they only have to be consistent across `setup_motors`, `calibrate`, `teleoperate`, and `record` so that the calibration JSONs in `calibration/` are picked up correctly.

`./scripts/setup.sh` runs `uv sync` and validates that every variable in `.env` is set (any value of `...` or empty triggers a failure).

## Hardware workflow

These scripts are run from the repo root and assume the venv is active (`source .venv/bin/activate`). Run them in this order the first time you connect the arms.

### 1. One-time motor setup

Assigns motor IDs and baud on each chain. Only needed once per arm, or after a motor swap.

```bash
./scripts/setup_motors.sh
```

### 2. Calibrate

Records each motor's homing offset and joint limits into `calibration/<id>.json`. Re-run if you re-clock a servo horn or otherwise change the mechanical zero.

```bash
./scripts/calibrate_so101.sh
```

> **Heads-up:** calibration values are stored in motor EEPROM. If `lerobot-teleoperate` reports *"Mismatch between calibration values in the motor and the calibration file"*, the on-motor values diverged from the JSON — pressing **ENTER** pushes the file back to the motors, while pressing **`c`** triggers a fresh calibration (which will invalidate any policy trained on the previous frame).

### 3. Teleoperate

Reads the leader arm and mirrors it on the follower at 60 Hz, with the front camera streamed to [Rerun](https://www.rerun.io/) for live visualization.

```bash
./scripts/run_teleoperation.sh
```

If you see `OpenCVCamera(0) latest frame is too old` or repeated `read failed (status=False)` messages: that's a USB/camera issue (cable, hub, macOS Continuity Camera grabbing the device, etc.), not a robot issue. Try MJPG (`fourcc: MJPG`), an external USB webcam, or a different USB port.

### 4. Record a dataset

Records `<NUM_EPISODES>` teleop episodes into a LeRobot dataset and (by default) pushes them to the Hugging Face Hub.

```bash
./scripts/record.sh <REPO_ID> <NUM_EPISODES> "<TASK_DESCRIPTION>" <RESUME>
# e.g.
./scripts/record.sh YOUR_USERNAME/towel-fold 30 "Fold the towel" true
```

All four arguments are optional — see `scripts/record.sh` for defaults.

## Policy training

Both training commands assume:
- A LeRobot dataset already exists on the Hub (`--dataset.repo_id=...`).
- You are running on a machine with a CUDA GPU (locally, in a Brev shell, or on any other CUDA host).
- `HF_TOKEN` is exported (or you have run `huggingface-cli login`) so weights and configs can be pushed at the end.

### Diffusion policy

Vanilla LeRobot diffusion policy. A safe starting point on ~30–100 demos.

```bash
lerobot-train \
  --dataset.repo_id=YOUR_USERNAME/<DATASET_NAME> \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_towel_fold_100 \
  --job_name=diffusion_towel_fold_100 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=YOUR_USERNAME/<POLICY_NAME> \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=1000 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=10 \
  --policy.prediction_type=epsilon \
  --policy.dropout=0.1
```

Notes on the choices above:
- `batch_size=64` — diffusion benefits from larger batches; safe on a 24 GB GPU.
- `steps=20000` — diffusion converges more slowly per step than ACT.
- `horizon=16` ≈ 0.5 s of actions @ 30 Hz; `n_action_steps=8` executes half the horizon each tick (open-loop chunk).
- `num_train_timesteps=100` / `num_inference_steps=10` — DDIM-style cheap rollouts at test time.
- `dropout=0.1` — light regularizer for small (≤100-demo) datasets.

### Multi-task DiT (diffusion transformer with CLIP conditioning)

Diffusion-transformer policy with a CLIP image and CLIP text encoder, RoPE, and image augmentations. Heavier than the vanilla diffusion policy — recommended only when you have a beefy GPU and at least a few hundred demos, or when the policy needs to ingest task strings.

```bash
lerobot-train \
  --dataset.repo_id=YOUR_USERNAME/towel-fold \
  --output_dir=./outputs/towel_fold_dit_gray \
  --batch_size=32 \
  --steps=40000 \
  --save_freq=2500 \
  --log_freq=100 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  --policy.n_obs_steps=2 \
  --policy.objective=diffusion \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.num_layers=8 \
  --policy.hidden_dim=512 \
  --policy.num_heads=8 \
  --policy.dropout=0.1 \
  --policy.use_rope=true \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.text_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape=[240,240] \
  --policy.image_crop_shape=[224,224] \
  --policy.image_crop_is_random=true \
  --policy.optimizer_lr=1e-4 \
  --policy.vision_encoder_lr_multiplier=0.1 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=5 \
  --dataset.image_transforms.tfs='{"desaturate":{"type":"ColorJitter","weight":10.0,"kwargs":{"saturation":[0.0,0.0]}},"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"rotation":{"type":"RandomRotation","kwargs":{"degrees":[-8,8]}},"translation":{"type":"RandomAffine","kwargs":{"degrees":0,"translate":[0.12,0.12]}}}' \
  --policy.repo_id=YOUR_USERNAME/towel-fold-dit-gray \
  --wandb.enable=true
```

Key knobs:
- `--policy.type=multi_task_dit` — DiT backbone with CLIP conditioning. Replaces the U-Net used by `--policy.type=diffusion`.
- `--policy.horizon=32`, `--policy.n_action_steps=24`, `--policy.n_obs_steps=2` — predict 32 future actions, execute 24 of them per tick, condition on the last 2 observations.
- `--policy.num_layers=8`, `--policy.hidden_dim=512`, `--policy.num_heads=8` — DiT capacity. Reduce if you run out of GPU memory.
- `--policy.use_rope=true` — rotary position embeddings inside the DiT.
- `--policy.vision_encoder_name` / `--policy.text_encoder_name` — both default to `openai/clip-vit-base-patch16` so a single CLIP checkpoint provides aligned image/text features. The vision encoder is fine-tuned at 0.1× the base LR (`--policy.vision_encoder_lr_multiplier`).
- `--policy.image_resize_shape=[240,240]` then `--policy.image_crop_shape=[224,224]` with `image_crop_is_random=true` — standard CLIP-style augmentation: resize, random-crop at training time (center-crop at eval).
- `--dataset.image_transforms.tfs='{...}'` — domain-randomization stack:
  - `desaturate` (weight 10) collapses saturation to 0 → effectively trains on a grayscale variant most of the time (hence the `_gray` in `output_dir`). Helps the model not overfit to towel colors.
  - `brightness`, `contrast`, `sharpness` — photometric jitter.
  - `rotation` (±8°) and `translation` (±12% of frame) — light geometric jitter.
  - `max_num_transforms=5` caps how many of the above are applied per sample.
- `--wandb.enable=true` — logs to W&B; needs `WANDB_API_KEY` exported (or `wandb login`).

The trained checkpoint is pushed to `--policy.repo_id` at the end and can be loaded by the async inference client below via `--pretrained_name_or_path`.

## Async inference

LeRobot's `policy_server` runs the model on a GPU host; the local `robot_client` connects over gRPC, streams observations to the server, and applies the returned action chunks on the SO-101.

### Server (e.g. on a Brev GPU shell)

```bash
brev shell <your-brev-instance>
git clone git@github.com:<your-username>/diffusing.git
cd diffusing
./scripts/setup.sh
source $HOME/.local/bin/env
uv pip install "lerobot[act, async]" grpcio grpcio-tools protobuf

uv run python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080
```

### Client (local machine with the SO-101)

Forward the server port locally:

```bash
brev port-forward <your-brev-instance> --port 8080:8080
```

Install the matching extras and start the client:

```bash
uv pip install "lerobot[act, async]" grpcio grpcio-tools protobuf

python -m lerobot.async_inference.robot_client \
  --server_address=localhost:8080 \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=$FOLLOWER_ID \
  --robot.calibration_dir=./calibration \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --policy_type=act \
  --pretrained_name_or_path=YOUR_USERNAME/<POLICY_NAME> \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=64 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True
```

Swap `--policy_type=act` for `--policy_type=diffusion` or `--policy_type=multi_task_dit` to run the corresponding policy.

## Repo layout

```
.
├── calibration/             # per-arm homing-offset JSONs (committed)
├── scripts/                 # thin wrappers over lerobot CLIs
│   ├── setup.sh             # uv install + .env validation
│   ├── setup_motors.sh      # one-time motor ID/baud setup
│   ├── calibrate_so101.sh   # per-arm calibration
│   ├── run_teleoperation.sh # leader → follower mirroring + camera
│   └── record.sh            # record a teleop dataset and push to the Hub
├── src/                     # project-specific Python (policies, etc.)
├── third_party/lerobot/     # vendored LeRobot submodule
└── outputs/                 # training outputs (gitignored)
```
