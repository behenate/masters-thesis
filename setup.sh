#!/usr/bin/env bash

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
JUPYTER_KERNEL_NAME="${JUPYTER_KERNEL_NAME:-thesis}"
JUPYTER_KERNEL_DISPLAY_NAME="${JUPYTER_KERNEL_DISPLAY_NAME:-Thesis}"

log_step() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

log_info() {
  printf '  - %s\n' "$1"
}

log_step "Step 1/4: Creating Python virtual environment"
if [ -f "${VENV_DIR}/bin/activate" ]; then
  log_info "Virtual environment already exists at ${VENV_DIR}; reusing it."
else
  python3 -m venv "$VENV_DIR"
  log_info "Created virtual environment at ${VENV_DIR}."
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log_info "Virtual environment activated from ${VENV_DIR}."

log_step "Step 2/4: Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

log_step "Step 3/4: Installing project and fine-tuning dependencies"
python -m pip install -e ".[lora-fine-tuning]" --no-build-isolation

log_step "Step 4/5: Registering Jupyter kernel"
python -m ipykernel install --user --name "${JUPYTER_KERNEL_NAME}" --display-name "${JUPYTER_KERNEL_DISPLAY_NAME}"

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  log_step "Step 5/5: Installing GPU-specific extras"
  python -m pip install packaging ninja
  python -m pip install flash-attn --no-build-isolation
else
  log_step "Step 5/5: Skipping GPU-specific extras"
  log_info "Set INSTALL_FLASH_ATTN=1 to install flash-attn and related build helpers."
fi

log_step "Setup complete"
log_info "Activate the environment with: source ${VENV_DIR}/bin/activate"
log_info "Notebook kernel registered as: ${JUPYTER_KERNEL_DISPLAY_NAME}"
