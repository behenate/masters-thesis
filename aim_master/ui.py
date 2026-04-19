from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, TypeVar

from prompt_toolkit.shortcuts import checkboxlist_dialog, radiolist_dialog, yes_no_dialog

T = TypeVar("T")


def _can_use_graphics() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys_platform_is_macos())


def sys_platform_is_macos() -> bool:
    import sys

    return sys.platform == "darwin"


def choose_directory_gui(title: str) -> Path | None:
    if not _can_use_graphics():
        return None

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    selected = filedialog.askdirectory(title=title, mustexist=True)
    root.destroy()

    if not selected:
        return None
    return Path(selected).expanduser().resolve()


def choose_one(
    title: str,
    text: str,
    options: Sequence[tuple[T, str]],
    default: T | None = None,
) -> T | None:
    if not options:
        return None

    try:
        return radiolist_dialog(title=title, text=text, values=options, default=default).run()
    except Exception:
        print(text)
        for index, (_, label) in enumerate(options, start=1):
            print(f"{index}. {label}")
        selected = input("Select an option by number (empty to cancel): ").strip()
        if not selected:
            return None
        return options[int(selected) - 1][0]


def choose_many(
    title: str,
    text: str,
    options: Sequence[tuple[T, str]],
    default_values: Sequence[T] | None = None,
) -> list[T]:
    if not options:
        return []

    try:
        result = checkboxlist_dialog(
            title=title,
            text=text,
            values=options,
            default_values=default_values or [],
        ).run()
        return list(result or [])
    except Exception:
        print(text)
        for index, (_, label) in enumerate(options, start=1):
            print(f"{index}. {label}")
        raw = input("Select comma-separated numbers (empty for none): ").strip()
        if not raw:
            return []
        selected_indexes = {int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()}
        return [value for index, (value, _) in enumerate(options, start=1) if index in selected_indexes]


def confirm(title: str, text: str, default: bool = True) -> bool:
    try:
        result = yes_no_dialog(title=title, text=text).run()
        return default if result is None else bool(result)
    except Exception:
        suffix = "[Y/n]" if default else "[y/N]"
        raw = input(f"{text} {suffix} ").strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}
