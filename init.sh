
set -euo pipefail

print_separator() {
    echo "--------------------------------"
}

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-N}"
    local reply=""

    if [[ "$default" == "Y" ]]; then
        read -r -p "$prompt [Y/n] " reply || true
        reply="${reply:-Y}"
    else
        read -r -p "$prompt [y/N] " reply || true
        reply="${reply:-N}"
    fi

    [[ "$reply" =~ ^[Yy]$ ]]
}

run_if_available() {
    local command_name="$1"
    shift

    if command -v "$command_name" >/dev/null 2>&1; then
        "$@"
    else
        echo "Command '$command_name' was not found. Install it first and rerun the test."
    fi
}

main() {
    if ! prompt_yes_no "Freesentry project in this folder will be initialized from scratch. Do you want to continue?" "N"; then
        exit 0
    fi

    echo "Preparing to initialize the project in this folder from scratch"
    echo "This may take a few minutes..."

    rm -rf .venv
    rm -f uv.lock pyproject.toml .python-version

    echo "Downloading uv"
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install uv." >&2
        exit 1
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if [[ -f "$HOME/.local/bin/env" ]]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not available on PATH after installation." >&2
        exit 1
    fi

    echo "Initializing python environment"
    uv init
    uv venv

    if [[ ! -f .venv/bin/activate ]]; then
        uv venv
    fi

    # shellcheck disable=SC1091
    . .venv/bin/activate

    echo "--------------------------------"
    echo "Current user: $(id -un)"
    echo "Installing dependencies..."
    echo "--------------------------------"

    echo "Insatlling fastapi"
    uv add fastapi --extra standard
    echo "Insatlling uvicorn"
    uv add uvicorn --extra standard
    echo "Insatlling gpiozero pigpio"
    uv add gpiozero pigpio
    echo "Insatlling numpy"
    uv add numpy
    echo "Insatlling opencv-python"
    uv add opencv-python
    echo "Insatlling opencv-python-headless"
    uv pip install "opencv-python-headless"
    echo "Insatlling ultralytics"
    uv add ultralytics
    echo "Insatlling debugpy"
    uv add debugpy
    echo "Insatlling picamera2"
    uv add picamera2
    echo "Insatlling python-periphery"
    uv pip install python-periphery

    echo "Current user: $(id -un)"

    print_separator
    echo "If you have AMD, Intel, NVidia or other GPU,"
    echo "you can use the Vulkan driver to run the app."
    echo "Download the Vulkan driver for your GPU"
    echo "from the vendor's website and install it."
    echo "NVidia: https://developer.nvidia.com/vulkan-driver"
    echo "Nvidia: https://www.google.com/search?q=nvidia+vulkan+driver+download"
    echo "AMD: https://www.google.com/search?q=amd+vulkan+driver+download"
    echo "Intel: https://www.google.com/search?q=intel+vulkan+driver+download"
    echo "Then run the following command to test if"
    echo "the Vulkan driver is installed correctly:"
    echo "vulkaninfo"
    print_separator

    if prompt_yes_no "Should I run the vulkaninfo command to test if the Vulkan driver is installed correctly?" "N"; then
        echo "Running vulkaninfo command"
        run_if_available "vulkaninfo" vulkaninfo
    else
        echo "OK, skipping vulkaninfo command"
    fi

    print_separator
    echo "If you have a NVIDIA GPU with CUDA support,"
    echo "you can use the CUDA version of this app."
    echo "To install the CUDA drivers download them from"
    echo "https://developer.nvidia.com/cuda-downloads"
    echo "and install the drivers and cuda toolkit."
    echo "Then run the following command to test if"
    echo "the CUDA drivers are installed correctly:"
    echo "nvcc --version"
    print_separator

    if prompt_yes_no "Should I run the nvcc --version command to test if the CUDA drivers are installed correctly?" "N"; then
        echo "Running nvcc --version command"
        run_if_available "nvcc" nvcc --version
    else
        echo "OK, skipping nvcc --version command"
    fi

    print_separator
    if ! prompt_yes_no "Do you want to setup the app to use a NVIDIA GPU with CUDA support?" "N"; then
        echo "Installing CPU only version of torch"
        uv add torch torchvision torchaudio
    else
        local torch_index_url=""

        echo "Installing CUDA version of torch"
        print_separator

        if prompt_yes_no "Do you want to install CUDA 13.X (Recommended - say Y), or CUDA 12.X (say N)?" "Y"; then
            echo "Installing CUDA 130"
            torch_index_url="https://download.pytorch.org/whl/cu130"
        else
            echo "Installing CUDA 124"
            torch_index_url="https://download.pytorch.org/whl/cu124"
        fi

        uv add torch torchvision torchaudio

        uv pip uninstall torch >/dev/null 2>&1 || true
        uv pip uninstall torchvision >/dev/null 2>&1 || true
        uv pip uninstall torchaudio >/dev/null 2>&1 || true

        uv pip install torch --index-url "$torch_index_url"
        uv pip install torchvision --index-url "$torch_index_url"
        uv pip install torchaudio --index-url "$torch_index_url"

        print_separator
        echo "Support for CUDA installed."
        echo "Now I will test if the CUDA is available."
        sleep 10
        echo "Testing if the CUDA drivers are installed correctly"
        echo "this may take a few minutes..."
        python -c "import torch; print(f'CUDA is available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
    fi

    print_separator
    echo "Environment initialized successfully"
    print_separator
    read -r -p "Press Enter to continue..." _
}

main "$@"
