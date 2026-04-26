# diffusing

SO-101 imitation learning with async inference support via [LeRobot](https://github.com/huggingface/lerobot).

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

## Modifying LeRobot

The `third_party/lerobot` directory is a git submodule. To make changes cleanly:

1. **Fork** [`huggingface/lerobot`](https://github.com/huggingface/lerobot) on GitHub.
2. Update `.gitmodules` to point to your fork:
   ```
   [submodule "third_party/lerobot"]
       path = third_party/lerobot
       url = https://github.com/<your-username>/lerobot.git
       branch = <your-branch>
   ```
3. Re-point the submodule and pull:
   ```bash
   git submodule sync
   git submodule update --remote
   ```
4. Make changes inside `third_party/lerobot`, commit and push to your fork.
5. Back in `diffusing`, commit the updated submodule pointer:
   ```bash
   git add third_party/lerobot
   git commit -m "Update lerobot submodule"
   ```

To pull upstream LeRobot changes into your fork, rebase your branch onto upstream main:

```bash
cd third_party/lerobot
git fetch upstream
git rebase upstream/main
```
