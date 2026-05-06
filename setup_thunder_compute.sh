#!/usr/bin/env bash

set -euo pipefail

CUDA_DOWNLOAD_URL="https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_TOOLKIT_PACKAGE="cuda-toolkit-13-2"
NGROK_BASE_URL="https://bin.equinox.io/c/bNyj1mQVY4c"
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

install_ngrok() {
  local arch
  local ngrok_archive
  local ngrok_url

  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64)
      ngrok_archive="ngrok-v3-stable-linux-amd64.tgz"
      ;;
    aarch64|arm64)
      ngrok_archive="ngrok-v3-stable-linux-arm64.tgz"
      ;;
    *)
      log_info "Unsupported architecture for automatic ngrok install: ${arch}"
      log_info "Install manually from https://ngrok.com/download"
      exit 1
      ;;
  esac

  ngrok_url="${NGROK_BASE_URL}/${ngrok_archive}"

  log_info "Downloading ngrok from ${ngrok_url}."
  wget -O "${ngrok_archive}" "${ngrok_url}"

  log_info "Installing ngrok to /usr/local/bin."
  sudo tar xvzf "${ngrok_archive}" -C /usr/local/bin
  rm -f "${ngrok_archive}"
}

log_step "Step 1/6: Checking CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
  log_info "CUDA toolkit already available."
else
  log_info "CUDA toolkit not found."
  log_info "Attempting automatic installation for Ubuntu 22.04."
  install_cuda_toolkit
fi

log_step "Step 2/6: Ensuring CUDA paths are in ${BASHRC_FILE}"
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

if command -v nvcc >/dev/null 2>&1; then
  log_info "CUDA toolkit setup completed."
else
  log_info "CUDA toolkit still not available after setup attempt."
  log_info "Check the NVIDIA instructions at:"
  log_info "${CUDA_DOWNLOAD_URL}"
  exit 1
fi

log_step "Step 3/6: Verifying CUDA compiler"
nvcc --version

log_step "Step 4/6: Checking ngrok"
if command -v ngrok >/dev/null 2>&1; then
  log_info "ngrok already available."
else
  install_ngrok
fi

log_info "ngrok setup completed."
ngrok version

log_step "Step 5/6: Running Python environment setup (skips flash attention)"
INSTALL_FLASH_ATTN=0 "${SCRIPT_DIR}/setup.sh"

log_step "Step 6/6: Final reminder"
log_info "If your notebook or shell was already open, restart it before running the workload."
