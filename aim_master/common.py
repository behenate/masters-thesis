from __future__ import annotations

import json
import logging
import re
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any

from aim import Repo
from aim.ext.cleanup import AutoClean, RobustExec

PACKAGE_DIR = Path(__file__).resolve().parent
MASTER_REPO_DIR = PACKAGE_DIR / "master_repo"
EXPORTS_DIR = PACKAGE_DIR / "exports"
MANIFEST_PATH = PACKAGE_DIR / "master_repo_manifest.json"

DEFAULT_METRIC_NAMES = {
    "loss",
    "learning_rate",
    "grad_norm",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "cross_entropy",
    "brier_score",
    "matthews_corrcoef",
    "__system__cpu",
    "__system__memory_percent",
}

NOISY_AIM_CLEANUP_PATTERN = re.compile(
    r"Exception ignored in atexit callback: <function AutoClean\.cleanup.*?"
    r"RuntimeError: can't create new thread at interpreter shutdown\n?",
    re.DOTALL,
)


def install_aim_cleanup_patch() -> None:
    original_cleanup = AutoClean.cleanup
    if getattr(original_cleanup, "__name__", "") == "_quiet_cleanup":
        return

    def _quiet_cleanup() -> None:
        logger = logging.getLogger(__name__)
        logger.debug("Cleaning up Aim resources.")
        try:
            example = RobustExec(stop_signal=AutoClean.stop_signal, target=AutoClean._cleanup)
            example.start()
            example.join()
        except RuntimeError as exc:
            if "can't create new thread at interpreter shutdown" not in str(exc):
                raise
            try:
                AutoClean._cleanup()
            except Exception:
                logger.debug("Aim cleanup fallback also failed during interpreter shutdown.", exc_info=True)

    AutoClean.cleanup = staticmethod(_quiet_cleanup)


def strip_aim_cleanup_noise(text: str) -> str:
    return NOISY_AIM_CLEANUP_PATTERN.sub("", text)


def run_command_filtered(
    command: list[str],
    *,
    check: bool = True,
    emit_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    clean_stdout = strip_aim_cleanup_noise(result.stdout)
    clean_stderr = strip_aim_cleanup_noise(result.stderr)

    if emit_output and clean_stdout:
        print(clean_stdout, end="")
    if emit_output and clean_stderr:
        print(clean_stderr, end="", file=sys.stderr)

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=clean_stdout,
            stderr=clean_stderr,
        )

    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=clean_stdout,
        stderr=clean_stderr,
    )


install_aim_cleanup_patch()


def ensure_master_repo_initialized(repo_path: Path | None = None) -> Path:
    target = Path(repo_path or MASTER_REPO_DIR).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    Repo(str(target), init=True)
    return target


def ensure_exports_dir(path: Path | None = None) -> Path:
    target = Path(path or EXPORTS_DIR).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_aim_cli_bin() -> str:
    candidates = [
        shutil.which("aim"),
        str(Path(sys.executable).parent / "aim"),
        str(Path(sys.argv[0]).parent / "aim"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Could not find the Aim CLI binary.")


def list_run_hashes_via_cli(repo_path: Path | str) -> list[str]:
    normalized_repo = normalize_repo_path(repo_path)
    command = [
        get_aim_cli_bin(),
        "runs",
        "--repo",
        str(normalized_repo),
        "ls",
    ]
    result = run_command_filtered(command, check=True, emit_output=False)
    hashes: list[str] = []
    hash_pattern = re.compile(r"^[0-9a-f]{24,}$")
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Total "):
            continue
        for token in stripped.split():
            if hash_pattern.fullmatch(token):
                hashes.append(token)
    return hashes


def load_manifest(path: Path | None = None) -> dict[str, dict[str, Any]]:
    target = Path(path or MANIFEST_PATH)
    if not target.exists():
        return {}
    return json.loads(target.read_text())


def save_manifest(data: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    target = Path(path or MANIFEST_PATH)
    target.write_text(json.dumps(data, indent=2, sort_keys=True))
    return target


def normalize_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser().resolve()
    if path.name == ".aim" and path.is_dir():
        return path.parent
    if (path / ".aim").is_dir():
        return path
    raise ValueError(f"{path} is not an Aim repository root and does not contain a .aim directory.")


def context_to_dict(context: Any) -> dict[str, Any]:
    if context is None:
        return {}
    if hasattr(context, "to_dict"):
        return dict(context.to_dict())
    if isinstance(context, dict):
        return dict(context)
    return {}


def context_to_label(context: Any) -> str:
    context_dict = context_to_dict(context)
    if not context_dict:
        return "no-context"
    return ", ".join(f"{key}={context_dict[key]}" for key in sorted(context_dict))


def metric_identity(metric_name: str, context: Any) -> str:
    context_dict = context_to_dict(context)
    serialized_context = json.dumps(context_dict, sort_keys=True, ensure_ascii=True)
    return f"{metric_name}::{serialized_context}"


def flatten_mapping(prefix: str, value: Any) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, inner_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_mapping(child_prefix, inner_value))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened[prefix] = json.dumps(value, ensure_ascii=False)
        return flattened
    flattened[prefix] = value
    return flattened
