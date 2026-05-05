#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

curl -LsSf https://astral.sh/uv/install.sh | sh

git submodule update --init --recursive

export PATH="$HOME/.local/bin:$PATH"

uv --version

uv sync --python 3.12 --extra robot

# ── Activate virtual environment ──────────────────────────────────────────────
VENV_PATH="$ROOT_DIR/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: virtual environment not found at $VENV_PATH"
    echo "       'uv sync' may have failed above."
    exit 1
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "  Active Python: $(which python) ($(python --version))"

# ── Check .env variables ──────────────────────────────────────────────────────
ENV_FILE="$ROOT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Checking .env variables..."
missing=0

while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    var_name="${line%%=*}"
    var_value="${line#*=}"

    # Fail if value is empty or literally "..."
    if [ -z "$var_value" ] || [ "$var_value" = "..." ]; then
        echo "  MISSING: $var_name"
        missing=$((missing + 1))
    else
        echo "  OK:      $var_name"
    fi
done < "$ENV_FILE"

if [ "$missing" -gt 0 ]; then
    echo ""
    echo "ERROR: $missing variable(s) are not set in .env — replace '...' with real values."
    exit 1
fi

echo ""
echo "All .env variables are set."
echo ""
echo "Setup complete. Environment is active for this shell session."
echo "To activate later, run: source $VENV_PATH/bin/activate"