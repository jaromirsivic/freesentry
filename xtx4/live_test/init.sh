#!/usr/bin/env bash
# XTX4 live test - initialise the Python virtual environment with uv (Linux / macOS).
# Usage: ./init.sh            (interactive)
#        ./init.sh --auto     (no questions; used by start.sh)
set -euo pipefail
cd "$(dirname "$0")"

AUTO=0
[[ "${1:-}" == "--auto" ]] && AUTO=1
PYTHON_VERSION="3.12"
VENV_PY=".venv/bin/python"

separator() { echo "--------------------------------"; }

ask_yes_no() {
    local reply=""
    read -r -p "$1 [y/N] " reply || true
    [[ "$reply" =~ ^[Yy]$ ]]
}

separator
echo "XTX4 live test - environment setup"
separator

# ---- uv ----------------------------------------------------------------------
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found - installing uv from astral.sh ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "ERROR: curl or wget is required to install uv." >&2
        exit 1
    fi
    if [[ -f "$HOME/.local/bin/env" ]]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    fi
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is still not available. Install it manually: https://docs.astral.sh/uv/" >&2
    exit 1
fi
uv --version

# ---- virtual environment -------------------------------------------------------
if [[ -x "$VENV_PY" ]]; then
    if [[ "$AUTO" == "0" ]] && ask_yes_no "A virtual environment already exists. Recreate it from scratch? (No = only update packages)"; then
        echo "Removing the existing environment..."
        rm -rf .venv
    fi
fi

if [[ ! -x "$VENV_PY" ]]; then
    echo "Creating virtual environment (Python $PYTHON_VERSION, downloaded by uv if needed)..."
    uv venv --no-project --python "$PYTHON_VERSION" .venv
fi

echo "Installing packages from requirements.txt ..."
uv pip install --python "$VENV_PY" -r requirements.txt

# ---- verification ------------------------------------------------------------
echo "Verifying the installation..."
if ! "$VENV_PY" -c "import ncnn, cv2, numpy, psutil; print('ncnn', ncnn.__version__, '| opencv', cv2.__version__, '| numpy', numpy.__version__)"; then
    echo "ERROR: verification failed - see the message above." >&2
    if [[ "$(uname -s)" == "Linux" ]]; then
        echo "Hint: OpenCV needs libGL / glib on minimal systems, e.g.: sudo apt install libgl1 libglib2.0-0" >&2
    fi
    exit 1
fi
if ! "$VENV_PY" -c "import sys; sys.path.insert(0, '..'); import xtx4.runtime; assert 'torch' not in sys.modules; print('xtx4.runtime OK (torch-free)')"; then
    echo "ERROR: the xtx4 package was not found next to live_test/ - keep live_test inside the xtx4 repository." >&2
    exit 1
fi
if ! "$VENV_PY" -c "import tkinter" >/dev/null 2>&1; then
    echo "Note: tkinter is unavailable - the app will use a console menu instead of the selection dialog."
fi
"$VENV_PY" -c "import ncnn; n = ncnn.get_gpu_count(); print('Vulkan GPUs usable by ncnn:', n); ncnn.destroy_gpu_instance()" 2>/dev/null || true

case "$(uname -s)" in
    Linux)
        if ! id -nG 2>/dev/null | grep -qw video; then
            echo "Note: your user is not in the 'video' group; if the camera cannot be opened run:"
            echo "      sudo usermod -aG video \$USER   (then log out and back in)"
        fi
        ;;
    Darwin)
        echo "Note: on first start macOS asks for camera access for your terminal app."
        echo "      If it was denied: System Settings -> Privacy & Security -> Camera."
        ;;
esac

separator
echo "Environment ready. Run ./start.sh to launch the live test."
separator
