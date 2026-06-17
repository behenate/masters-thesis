from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import joblib

from common import ARTIFACT_PATHS, BENCHMARK_DIR, ROOT, ensure_directories, load_module, write_json


LOCAL_HF_CACHE = BENCHMARK_DIR / ".hf_cache"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(LOCAL_HF_CACHE / "datasets"))


def artifact_metadata_path(method: str) -> Path:
    return ARTIFACT_PATHS[method].with_suffix(ARTIFACT_PATHS[method].suffix + ".metadata.json")


def prepare_naive_bayes() -> dict[str, Any]:
    module = load_module(
        ROOT / "sota" / "tfidf_naive_bayes" / "run_tfidf_naive_bayes.py",
        "deployment_tfidf_naive_bayes",
    )
    args = module.build_parser().parse_args([])
    model, _, metadata, training_seconds = module.train_model(args)
    joblib.dump(model, ARTIFACT_PATHS["tfidf_naive_bayes"], compress=0)
    return {"training_seconds": training_seconds, **metadata}


def prepare_logistic_regression() -> dict[str, Any]:
    method_dir = ROOT / "sota" / "tfidf_logistic_regression"
    artifact = joblib.load(method_dir / "tuned_model.joblib")
    joblib.dump(artifact, ARTIFACT_PATHS["tfidf_logistic_regression"], compress=0)
    return json.loads((method_dir / "tuned_config.json").read_text(encoding="utf-8"))


def prepare_linear_svm() -> dict[str, Any]:
    method_dir = ROOT / "sota" / "tfidf_linear_svm"
    artifact = joblib.load(method_dir / "tuned_model.joblib")
    joblib.dump(artifact, ARTIFACT_PATHS["tfidf_linear_svm"], compress=0)
    return json.loads((method_dir / "tuned_config.json").read_text(encoding="utf-8"))


def prepare_fasttext() -> dict[str, Any]:
    method_dir = ROOT / "sota" / "fasttext"
    shutil.copy2(method_dir / "tuned_model.bin", ARTIFACT_PATHS["fasttext"])
    return json.loads((method_dir / "tuned_config.json").read_text(encoding="utf-8"))


def prepare_minilm() -> dict[str, Any]:
    method_dir = ROOT / "sota" / "minilm_logistic_regression"
    artifact = joblib.load(method_dir / "tuned_classifier.joblib")
    joblib.dump(artifact, ARTIFACT_PATHS["minilm_logistic_regression"], compress=0)
    return json.loads((method_dir / "tuned_config.json").read_text(encoding="utf-8"))


PREPARERS = {
    "tfidf_naive_bayes": prepare_naive_bayes,
    "tfidf_logistic_regression": prepare_logistic_regression,
    "tfidf_linear_svm": prepare_linear_svm,
    "fasttext": prepare_fasttext,
    "minilm_logistic_regression": prepare_minilm,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recreate and persist one deployment benchmark artifact.")
    parser.add_argument("--method", required=True, choices=sorted(PREPARERS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    artifact = ARTIFACT_PATHS[args.method]
    if artifact.exists() and not args.force:
        print(f"artifact_exists={artifact}", flush=True)
        return 0

    print(f"preparing={args.method}", flush=True)
    started = time.perf_counter()
    metadata = PREPARERS[args.method]()
    payload = {
        "method": args.method,
        "artifact_path": str(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "total_prepare_seconds": time.perf_counter() - started,
        **metadata,
    }
    write_json(artifact_metadata_path(args.method), payload)
    print(f"artifact={artifact}", flush=True)
    print(f"size_bytes={artifact.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
