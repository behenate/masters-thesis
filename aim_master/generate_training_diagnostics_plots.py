from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from aim_master.common import load_manifest
from aim_master.export_metrics import load_run_from_source


ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = ROOT / "aim_master" / "exports" / "runs_16_20260610_224430.csv"
PLOTS_DIR = ROOT / "thesis" / "tex" / "plots"

SELECTED_CONFIGS = ["K01", "K04", "K08", "K11", "K15"]
CONFIG_COLORS = {
    "K01": "plotBlue",
    "K04": "plotSpam",
    "K08": "plotHam",
    "K11": "plotAccent",
    "K15": "black!70",
}
CONFIG_MARKS = {
    "K01": "*",
    "K04": "square*",
    "K08": "triangle*",
    "K11": "diamond*",
    "K15": "pentagon*",
}


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def parse_config(run_name: str) -> dict[str, Any]:
    match = re.search(r"__(\d+)__([^ ]+)", run_name)
    if not match:
        raise ValueError(f"Could not parse config from run name: {run_name}")
    config_index = int(match.group(1))
    config_id = match.group(2)
    seq_match = re.search(r"seq(\d+)", config_id)
    if not seq_match:
        raise ValueError(f"Could not parse sequence length from run name: {run_name}")
    return {
        "config_index": config_index,
        "config_label": f"K{config_index:02d}",
        "config_id": config_id,
        "max_seq_length": int(seq_match.group(1)),
    }


def load_run_metadata(run_hashes: list[str]) -> pd.DataFrame:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for run_hash in run_hashes:
        run, source_repo, _ = load_run_from_source(run_hash, manifest)
        parsed = parse_config(str(run.name))
        rows.append(
            {
                "run_hash": run_hash,
                "run_name": str(run.name),
                "source_repo": str(source_repo),
                **parsed,
            }
        )
    return pd.DataFrame(rows)


def with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata = load_run_metadata(sorted(df["run_hash"].unique()))
    merged = df.merge(metadata, on="run_hash", how="left")
    max_steps = (
        merged[merged["metric_name"].eq("loss") & merged["context_label"].eq("subset=train")]
        .groupby("run_hash")["step"]
        .max()
        .rename("max_train_step")
    )
    merged = merged.merge(max_steps, on="run_hash", how="left")
    merged["training_progress"] = merged["step"] / merged["max_train_step"] * 100.0
    return merged


def sample_curve(df: pd.DataFrame, max_points: int = 80) -> pd.DataFrame:
    df = df.sort_values("training_progress")
    if len(df) <= max_points:
        return df
    indexes = sorted({round(i * (len(df) - 1) / (max_points - 1)) for i in range(max_points)})
    return df.iloc[indexes]


def coordinates(rows: pd.DataFrame, *, pct: bool = False, sample: bool = False, min_value: float | None = None) -> str:
    if sample:
        rows = sample_curve(rows)
    points = []
    for _, row in rows.sort_values("training_progress").iterrows():
        value = float(row["value"]) * 100.0 if pct else float(row["value"])
        if min_value is not None:
            value = max(value, min_value)
        points.append(f"({fmt(float(row['training_progress']), 1)},{fmt(value, 5)})")
    return " ".join(points)


def legend_for_selected() -> str:
    parts = []
    for config in SELECTED_CONFIGS:
        parts.append(
            rf"\tikz[baseline=-0.5ex]\draw[{CONFIG_COLORS[config]}, line width=0.8pt] "
            rf"(0,0) -- (0.45,0);~\texttt{{{config}}}"
        )
    return r"{\scriptsize " + r"\quad ".join(parts) + r"}\\[0.25em]"


def axis_for_metric(
    df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    ymode_log: bool = False,
    ymin: float | None = None,
    ymax: float | None = None,
    pct: bool = False,
    sample: bool = False,
    min_value: float | None = None,
) -> str:
    options = [
        r"    width=0.455\textwidth,",
        r"    height=0.29\textwidth,",
        rf"    title={{{title}}},",
        r"    title style={font=\scriptsize, yshift=-1.2ex},",
        r"    xmin=0,",
        r"    xmax=100,",
        r"    xlabel={Postęp treningu [\%]},",
        rf"    ylabel={{{ylabel}}},",
        r"    xmajorgrids,",
        r"    ymajorgrids,",
        r"    grid style={plotGrid},",
        r"    tick style={draw=none},",
        r"    xticklabel style={font=\tiny},",
        r"    yticklabel style={font=\tiny},",
        r"    label style={font=\scriptsize},",
    ]
    if ymode_log:
        options.append(r"    ymode=log,")
    if ymin is not None:
        options.append(rf"    ymin={fmt(ymin, 4)},")
    if ymax is not None:
        options.append(rf"    ymax={fmt(ymax, 4)},")

    lines = [r"\begin{tikzpicture}", r"\begin{axis}["]
    lines.extend(options)
    lines.append(r"]")
    for config in SELECTED_CONFIGS:
        subset = df[df["config_label"].eq(config)]
        if subset.empty:
            continue
        lines.append(
            rf"\addplot+[mark=none, line width=0.75pt, color={CONFIG_COLORS[config]}] "
            rf"coordinates {{{coordinates(subset, pct=pct, sample=sample, min_value=min_value)}}};"
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    return "\n".join(lines)


def write_loss_diagnostics(df: pd.DataFrame) -> None:
    selected = df[df["config_label"].isin(SELECTED_CONFIGS)]
    train_loss = selected[selected["metric_name"].eq("loss") & selected["context_label"].eq("subset=train")]
    train_loss = train_loss.sort_values(["config_label", "training_progress"]).copy()
    train_loss["value"] = train_loss.groupby("config_label")["value"].transform(
        lambda values: values.rolling(window=7, center=True, min_periods=1).median()
    )
    validation_loss = selected[selected["metric_name"].eq("loss") & selected["context_label"].eq("subset=validation")]
    lines = [
        r"\begingroup",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        legend_for_selected(),
        r"\begin{tabular}{cc}",
        axis_for_metric(
            train_loss,
            title=r"\emph{training loss}",
            ylabel=r"Loss",
            ymode_log=True,
            ymin=0.00005,
            ymax=2.0,
            sample=True,
            min_value=0.00005,
        )
        + " & "
        + axis_for_metric(
            validation_loss,
            title=r"\emph{validation loss}",
            ylabel=r"Loss",
            ymode_log=True,
            ymin=0.002,
            ymax=0.6,
            min_value=0.00005,
        )
        + r" \\",
        r"\end{tabular}",
        r"\endgroup",
    ]
    write(PLOTS_DIR / "checkpoint_training_loss_diagnostics.tex", "\n".join(lines))


def write_validation_metric_diagnostics(df: pd.DataFrame) -> None:
    selected = df[df["config_label"].isin(SELECTED_CONFIGS)]
    validation_f1 = selected[selected["metric_name"].eq("f1") & selected["context_label"].eq("subset=validation")]
    validation_accuracy = selected[
        selected["metric_name"].eq("accuracy") & selected["context_label"].eq("subset=validation")
    ]
    lines = [
        r"\begingroup",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        legend_for_selected(),
        r"\begin{tabular}{cc}",
        axis_for_metric(
            validation_f1,
            title=r"F1 na zbiorze walidacyjnym",
            ylabel=r"F1 [\%]",
            ymin=0,
            ymax=101,
            pct=True,
        )
        + " & "
        + axis_for_metric(
            validation_accuracy,
            title=r"\emph{accuracy} na zbiorze walidacyjnym",
            ylabel=r"\emph{Accuracy} [\%]",
            ymin=40,
            ymax=101,
            pct=True,
        )
        + r" \\",
        r"\end{tabular}",
        r"\endgroup",
    ]
    write(PLOTS_DIR / "checkpoint_validation_metrics_diagnostics.tex", "\n".join(lines))


def write_runtime_context_comparison(df: pd.DataFrame) -> None:
    train = df[df["context_label"].eq("subset=train")]
    runtime = (
        train[train["metric_name"].isin(["train_runtime", "train_steps_per_second", "train_samples_per_second"])]
        .pivot_table(
            index=["run_hash", "config_index", "config_label", "config_id", "max_seq_length"],
            columns="metric_name",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    runtime = runtime.dropna(subset=["train_runtime"])
    runtime["runtime_h"] = runtime["train_runtime"] / 3600.0
    runtime = runtime.sort_values("config_index")
    runtime.to_csv(ROOT / "aim_master" / "exports" / "training_runtime_by_run.csv", index=False)

    grouped = (
        runtime.groupby("max_seq_length")
        .agg(
            n=("run_hash", "count"),
            mean_runtime_h=("runtime_h", "mean"),
            median_runtime_h=("runtime_h", "median"),
            min_runtime_h=("runtime_h", "min"),
            max_runtime_h=("runtime_h", "max"),
            mean_steps_s=("train_steps_per_second", "mean"),
            mean_samples_s=("train_samples_per_second", "mean"),
        )
        .reset_index()
        .sort_values("max_seq_length")
    )
    grouped.to_csv(ROOT / "aim_master" / "exports" / "training_runtime_by_context.csv", index=False)

    labels = ",".join(runtime["config_label"].tolist())
    xticks = ",".join(str(int(value)) for value in runtime["config_index"])
    coords_512 = " ".join(
        f"({int(row.config_index)},{fmt(row.runtime_h, 2)})"
        for _, row in runtime[runtime["max_seq_length"].eq(512)].iterrows()
    )
    coords_1024 = " ".join(
        f"({int(row.config_index)},{fmt(row.runtime_h, 2)})"
        for _, row in runtime[runtime["max_seq_length"].eq(1024)].iterrows()
    )
    max_runtime_h = runtime["runtime_h"].max()
    lines = [
        r"\begingroup",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.92\textwidth,",
        r"    height=0.36\textwidth,",
        r"    ybar,",
        r"    bar width=8pt,",
        rf"    xmin={int(runtime['config_index'].min()) - 0.6},",
        rf"    xmax={int(runtime['config_index'].max()) + 0.6},",
        rf"    xtick={{{xticks}}},",
        rf"    xticklabels={{{labels}}},",
        r"    ymin=0,",
        rf"    ymax={fmt(max_runtime_h * 1.18, 2)},",
        r"    ylabel={Czas [h]},",
        r"    xlabel={Konfiguracja},",
        r"    nodes near coords,",
        r"    nodes near coords style={font=\tiny},",
        r"    grid style={plotGrid},",
        r"    ymajorgrids,",
        r"    tick style={draw=none},",
        r"    xticklabel style={font=\scriptsize, rotate=45, anchor=east},",
        r"    yticklabel style={font=\footnotesize},",
        r"    label style={font=\small},",
        r"    legend style={font=\scriptsize, draw=none, fill=none, at={(0.02,0.98)}, anchor=north west},",
        r"    legend cell align={left},",
        r"]",
        rf"\addplot+[fill=plotBlue!72, draw=plotBlue!85!black, line width=0.25pt, bar shift=0pt] coordinates {{{coords_512}}};",
        rf"\addplot+[fill=plotSpam!72, draw=plotSpam!85!black, line width=0.25pt, bar shift=0pt] coordinates {{{coords_1024}}};",
        r"\legend{512 tokenów, 1024 tokeny}",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\endgroup",
    ]
    write(PLOTS_DIR / "checkpoint_training_runtime_context.tex", "\n".join(lines))


def main() -> int:
    df = pd.read_csv(EXPORT_PATH)
    df = with_metadata(df)
    write_loss_diagnostics(df)
    write_validation_metric_diagnostics(df)
    write_runtime_context_comparison(df)
    print(f"Loaded rows: {len(df)}")
    print(f"Runs: {df['run_hash'].nunique()}")
    print(f"Wrote plots to: {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
