#!/usr/bin/env bash

set -euo pipefail

CUDA_DOWNLOAD_URL="https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_TOOLKIT_PACKAGE="cuda-toolkit-13-2"
BASHRC_FILE="${HOME}/.bashrc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_step() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

log_info() {
  printf '  - %s\n' "$1"
}

ensure_line_in_file() {
  local file="$1"
  local line="$2"

  touch "$file"

  if ! grep -Fqx "$line" "$file"; then
    printf '%s\n' "$line" >> "$file"
    log_info "Added to ${file}: ${line}"
  else
    log_info "Already present in ${file}: ${line}"
  fi
}

install_cuda_toolkit() {
  log_info "Downloading CUDA keyring package."
  wget "${CUDA_KEYRING_URL}"

  log_info "Installing CUDA keyring package."
  sudo dpkg -i "${CUDA_KEYRING_DEB}"

  log_info "Refreshing apt package index."
  sudo apt-get update

  log_info "Installing ${CUDA_TOOLKIT_PACKAGE}."
  sudo apt-get -y install "${CUDA_TOOLKIT_PACKAGE}"
}

log_step "Step 1/5: Checking CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
  log_info "CUDA toolkit already available."
else
  log_info "CUDA toolkit not found."
  log_info "Attempting automatic installation for Ubuntu 22.04."
  install_cuda_toolkit

  if command -v nvcc >/dev/null 2>&1; then
    log_info "CUDA toolkit installation completed."
  else
    log_info "CUDA toolkit still not available after installation attempt."
    log_info "Check the NVIDIA instructions at:"
    log_info "${CUDA_DOWNLOAD_URL}"
    exit 1
  fi
fi

log_step "Step 2/5: Ensuring CUDA paths are in ${BASHRC_FILE}"
touch "$BASHRC_FILE"

if [ ! -w "$BASHRC_FILE" ]; then
  log_info "${BASHRC_FILE} is not writable, trying to fix permissions for the current user."
  chmod u+w "$BASHRC_FILE"
fi

ensure_line_in_file "$BASHRC_FILE" 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}'
ensure_line_in_file "$BASHRC_FILE" 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
log_info "CUDA paths exported for the current run."

log_step "Step 3/5: Verifying CUDA compiler"
nvcc --version

log_step "Step 4/5: Running Python environment setup"
INSTALL_FLASH_ATTN=1 "${SCRIPT_DIR}/setup.sh"

log_step "Step 5/5: Final reminder"
log_info "If your notebook or shell was already open, restart it before running the workload."
