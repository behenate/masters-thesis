#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
AIM_BIN="${AIM_BIN:-${VENV_DIR}/bin/aim}"
AIM_HOST="${AIM_HOST:-127.0.0.1}"
AIM_PORT="${AIM_PORT:-43800}"
AIM_REPO="${AIM_REPO:-${SCRIPT_DIR}}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
NGROK_BIN="${NGROK_BIN:-ngrok}"
NGROK_API_URL="${NGROK_API_URL:-http://127.0.0.1:4040/api/tunnels}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-30}"
RUNTIME_DIR="${RUNTIME_DIR:-${SCRIPT_DIR}/.aim-runtime}"
SHORTEN_PUBLIC_URL="${SHORTEN_PUBLIC_URL:-1}"

AIM_PID=""
NGROK_PID=""

log_step() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

log_info() {
  printf '  - %s\n' "$1"
}

load_env_file() {
  if [ -f "${ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    set -a
    source "${ENV_FILE}"
    set +a
  fi
}

cleanup() {
  local exit_code=$?

  if [ -n "${NGROK_PID}" ] && kill -0 "${NGROK_PID}" >/dev/null 2>&1; then
    kill "${NGROK_PID}" >/dev/null 2>&1 || true
  fi

  if [ -n "${AIM_PID}" ] && kill -0 "${AIM_PID}" >/dev/null 2>&1; then
    kill "${AIM_PID}" >/dev/null 2>&1 || true
  fi

  exit "${exit_code}"
}

wait_for_http() {
  local url="$1"
  local timeout="$2"
  local elapsed=0

  while [ "${elapsed}" -lt "${timeout}" ]; do
    if curl --silent --fail "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  return 1
}

read_ngrok_public_url() {
  python3 - "$1" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=5) as response:
    payload = json.load(response)

for tunnel in payload.get("tunnels", []):
    public_url = tunnel.get("public_url", "")
    if public_url.startswith("https://"):
        print(public_url)
        raise SystemExit(0)

for tunnel in payload.get("tunnels", []):
    public_url = tunnel.get("public_url", "")
    if public_url:
        print(public_url)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

shorten_public_url() {
  python3 - "$1" <<'PY'
import json
import sys
import subprocess

target_url = sys.argv[1]

def shorten_with_lnkua(url: str) -> str:
    command = [
        "curl",
        "-ksS",
        "-X",
        "POST",
        "https://lnk.ua/api/v1/link/create",
        "-H",
        "Accept: application/json",
        "-H",
        "Authorization: Bearer public",
        "-F",
        f"link={url}",
    ]
    response = subprocess.run(command, check=True, capture_output=True, text=True)
    body = json.loads(response.stdout)
    shortened = body.get("result", {}).get("lnk", "").strip()
    if not shortened.startswith(("http://", "https://")):
        raise RuntimeError(f"lnk.ua returned unexpected payload: {body}")
    return shortened

try:
    print(shorten_with_lnkua(target_url))
except Exception as exc:
    print(f"lnk.ua: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

print_failure_logs() {
  if [ -f "${RUNTIME_DIR}/aim.log" ]; then
    log_info "Aim log:"
    tail -n 20 "${RUNTIME_DIR}/aim.log" || true
  fi

  if [ -f "${RUNTIME_DIR}/ngrok.log" ]; then
    log_info "ngrok log:"
    tail -n 20 "${RUNTIME_DIR}/ngrok.log" || true
  fi
}

trap cleanup EXIT INT TERM

mkdir -p "${RUNTIME_DIR}"
load_env_file

log_step "Step 1/5: Checking required commands"
if [ ! -x "${AIM_BIN}" ]; then
  log_info "Aim executable not found at ${AIM_BIN}."
  log_info "Run ./setup.sh first or set AIM_BIN explicitly."
  exit 1
fi

if ! command -v "${NGROK_BIN}" >/dev/null 2>&1; then
  log_info "ngrok is not available on PATH."
  log_info "Install ngrok via Apt with the following commands:"
  log_info "curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
  log_info "echo \"deb https://ngrok-agent.s3.amazonaws.com bookworm main\" | sudo tee /etc/apt/sources.list.d/ngrok.list"
  log_info "sudo apt update"
  log_info "sudo apt install ngrok"
  log_info "Store your token in ${ENV_FILE} as NGROK_AUTHTOKEN=..."
  log_info "Then run: ngrok config add-authtoken \"\$NGROK_AUTHTOKEN\""
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  log_info "curl is required to detect when Aim and ngrok are ready."
  exit 1
fi

log_info "Aim binary: ${AIM_BIN}"
log_info "ngrok binary: $(command -v "${NGROK_BIN}")"

if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
  log_info "Configuring ngrok authtoken from ${ENV_FILE}."
  "${NGROK_BIN}" config add-authtoken "${NGROK_AUTHTOKEN}" >/dev/null
else
  log_info "NGROK_AUTHTOKEN is not set in ${ENV_FILE}; assuming ngrok is already configured."
fi

log_step "Step 2/5: Starting Aim dashboard"
"${AIM_BIN}" up \
  --host "${AIM_HOST}" \
  --port "${AIM_PORT}" \
  --repo "${AIM_REPO}" \
  --yes >"${RUNTIME_DIR}/aim.log" 2>&1 &
AIM_PID=$!

if ! wait_for_http "http://${AIM_HOST}:${AIM_PORT}" "${STARTUP_TIMEOUT_SECONDS}"; then
  log_info "Aim did not become ready within ${STARTUP_TIMEOUT_SECONDS} seconds."
  print_failure_logs
  exit 1
fi

log_info "Aim dashboard is responding at http://${AIM_HOST}:${AIM_PORT}"

log_step "Step 3/5: Starting ngrok tunnel"
"${NGROK_BIN}" http "${AIM_HOST}:${AIM_PORT}" >"${RUNTIME_DIR}/ngrok.log" 2>&1 &
NGROK_PID=$!

if ! wait_for_http "${NGROK_API_URL}" "${STARTUP_TIMEOUT_SECONDS}"; then
  log_info "ngrok API did not become ready within ${STARTUP_TIMEOUT_SECONDS} seconds."
  print_failure_logs
  exit 1
fi

log_step "Step 4/5: Reading public URL"
NGROK_PUBLIC_URL=""
for _ in $(seq 1 "${STARTUP_TIMEOUT_SECONDS}"); do
  if NGROK_PUBLIC_URL="$(read_ngrok_public_url "${NGROK_API_URL}" 2>/dev/null)"; then
    break
  fi
  sleep 1
done

if [ -z "${NGROK_PUBLIC_URL}" ]; then
  log_info "Could not find a public ngrok URL."
  print_failure_logs
  exit 1
fi

log_info "Local Aim dashboard: http://${AIM_HOST}:${AIM_PORT}"
log_info "Public ngrok URL: ${NGROK_PUBLIC_URL}"

if [ "${SHORTEN_PUBLIC_URL}" = "1" ]; then
  log_step "Step 5/6: Shortening public URL"
  SHORT_PUBLIC_URL=""
  SHORTENER_ERROR=""
  if SHORT_PUBLIC_URL="$(shorten_public_url "${NGROK_PUBLIC_URL}" 2>"${RUNTIME_DIR}/shortener.log")"; then
    log_info "Shortened URL: ${SHORT_PUBLIC_URL}"
  else
    log_info "URL shortening failed; keeping the original ngrok URL."
    SHORTENER_ERROR="$(tr '\n' ' ' <"${RUNTIME_DIR}/shortener.log" | sed 's/[[:space:]]\\+/ /g' | sed 's/^ //; s/ $//')"
    if [ -n "${SHORTENER_ERROR}" ]; then
      log_info "Shortener error: ${SHORTENER_ERROR}"
    fi
  fi
else
  log_step "Step 5/6: Skipping URL shortening"
  log_info "Set SHORTEN_PUBLIC_URL=1 to shorten the public link."
fi

log_step "Step 6/6: Keeping the tunnel alive"
log_info "Press Ctrl-C to stop Aim and close the ngrok tunnel."

while true; do
  if ! kill -0 "${AIM_PID}" >/dev/null 2>&1; then
    log_info "Aim exited unexpectedly."
    print_failure_logs
    exit 1
  fi

  if ! kill -0 "${NGROK_PID}" >/dev/null 2>&1; then
    log_info "ngrok exited unexpectedly."
    print_failure_logs
    exit 1
  fi

  sleep 2
done
