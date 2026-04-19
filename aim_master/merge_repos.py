from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from aim import Repo

from .common import (
    ensure_master_repo_initialized,
    get_aim_cli_bin,
    list_run_hashes_via_cli,
    load_manifest,
    normalize_repo_path,
    run_command_filtered,
    save_manifest,
)
from .ui import choose_directory_gui, confirm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely merge downloaded Aim repositories into a master repo.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Path to an Aim repository root or a .aim directory. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--master-repo",
        default=None,
        help="Override the default master repo path (defaults to aim_master/master_repo).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move runs instead of copying them.",
    )
    return parser.parse_args()


def collect_sources_interactively() -> list[Path]:
    sources: list[Path] = []

    while True:
        selected_path = choose_directory_gui("Select an Aim repository or a .aim directory to merge")
        if selected_path is None:
            if not sources:
                manual = input("Enter an Aim repo path manually (empty to cancel): ").strip()
                if not manual:
                    break
                selected_path = Path(manual).expanduser().resolve()
            else:
                break

        try:
            normalized = normalize_repo_path(selected_path)
        except ValueError as exc:
            print(f"Skipping invalid selection: {exc}")
            if not confirm("Try Again", "Pick another repository?", default=True):
                break
            continue

        if normalized not in sources:
            sources.append(normalized)
            print(f"Queued {normalized}")
        else:
            print(f"Already queued {normalized}")

        if not confirm("Add Another Repo", "Select another Aim repository to merge?", default=False):
            break

    return sources


def list_run_hashes(repo_path: Path) -> list[str]:
    repo = Repo(str(repo_path))
    return [run.hash for run in repo.iter_runs()]


def close_runs(repo_path: Path, run_hashes: list[str]) -> None:
    if not run_hashes:
        return
    command = [
        get_aim_cli_bin(),
        "runs",
        "--repo",
        str(repo_path),
        "close",
        "-y",
        *run_hashes,
    ]
    run_command_filtered(command, check=True)


def run_exists(repo_path: Path, run_hash: str) -> bool:
    return run_hash in set(list_run_hashes_via_cli(repo_path))


def merge_repo(source_repo: Path, destination_repo: Path, *, move: bool) -> tuple[list[str], list[str]]:
    run_hashes = list_run_hashes(source_repo)
    if not run_hashes:
        return [], []

    destination_before = set(list_run_hashes_via_cli(destination_repo))
    print(f"Closing {len(run_hashes)} source runs in {source_repo} to clear stale Aim locks")
    close_runs(source_repo, run_hashes)

    action = "mv" if move else "cp"

    for run_hash in run_hashes:
        print(f"Merging run {run_hash}")
        command = [
            get_aim_cli_bin(),
            "runs",
            "--repo",
            str(source_repo),
            action,
            run_hash,
            "--destination",
            str(destination_repo),
        ]
        run_command_filtered(command, check=True)

    destination_after = set(list_run_hashes_via_cli(destination_repo))
    merged_hashes = [run_hash for run_hash in run_hashes if run_hash in destination_after]
    failed_hashes = [run_hash for run_hash in run_hashes if run_hash not in destination_after]

    return merged_hashes, failed_hashes


def print_merge_summary(master_repo: Path) -> None:
    hashes = list_run_hashes_via_cli(master_repo)
    print(f"Master repo is ready at: {master_repo}")
    print(f"Total runs in master repo: {len(hashes)}")


def main() -> int:
    args = parse_args()
    master_repo = ensure_master_repo_initialized(Path(args.master_repo) if args.master_repo else None)

    provided_sources = [normalize_repo_path(path) for path in args.source]
    sources = provided_sources or collect_sources_interactively()

    if not sources:
        print("No source Aim repositories were selected.")
        return 0

    unique_sources: list[Path] = []
    for source in sources:
        if source == master_repo:
            print(f"Skipping {source}: this is already the master repo.")
            continue
        if source not in unique_sources:
            unique_sources.append(source)

    if not unique_sources:
        print("No valid source Aim repositories remain after filtering.")
        return 0

    print(f"Master repo: {master_repo}")
    print("Sources to merge:")
    for source in unique_sources:
        print(f"- {source}")

    if not confirm("Merge Runs", "Start merging the selected Aim repositories into the master repo?", default=True):
        print("Merge cancelled.")
        return 0

    all_failed_hashes: list[tuple[Path, list[str]]] = []
    manifest = load_manifest()

    for source in unique_sources:
        action = "Moving" if args.move else "Copying"
        print(f"{action} runs from {source} into {master_repo}")
        merged_hashes, failed_hashes = merge_repo(source, master_repo, move=args.move)
        for run_hash in merged_hashes:
            manifest[run_hash] = {
                "source_repo": str(source),
                "master_repo": str(master_repo),
            }
        if failed_hashes:
            all_failed_hashes.append((source, failed_hashes))

    save_manifest(manifest)

    if all_failed_hashes:
        print("Merge finished with failures.")
        for source, failed_hashes in all_failed_hashes:
            print(f"Failed from {source}:")
            for run_hash in failed_hashes:
                print(f"- {run_hash}")
        print_merge_summary(master_repo)
        return 1

    print("Merge complete.")
    print_merge_summary(master_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
