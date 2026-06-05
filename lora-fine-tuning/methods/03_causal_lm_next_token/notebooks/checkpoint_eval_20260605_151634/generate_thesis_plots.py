#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
SUMMARY_PATH = Path(__file__).resolve().parent / "summary.csv"
PLOTS_DIR = ROOT / "thesis" / "tex" / "plots"
TABLES_DIR = ROOT / "thesis" / "tex" / "tables"
RANKING_CSV = Path(__file__).resolve().parent / "checkpoint_config_ranking.csv"
METHOD_SUMMARY_PATHS = {
    "01": ROOT / "lora-fine-tuning" / "methods" / "01_sequence_classification" / "summary.csv",
    "02": ROOT / "lora-fine-tuning" / "methods" / "02_causal_lm_generation_parsing" / "summary.csv",
    "03": SUMMARY_PATH,
    "04": ROOT / "lora-fine-tuning" / "methods" / "04_causal_lm_structured_generation" / "summary.csv",
}
METHOD_LABELS = {
    "01": r"01",
    "02": r"02",
    "03": r"03",
    "04": r"04",
}
METHOD_NAMES = {
    "01": "Klasyfikacja sekwencji",
    "02": "Generowanie i parsowanie",
    "03": "Następny token",
    "04": "Odpowiedź strukturyzowana",
}
CONFIG_LABELS: dict[int, str] = {}

DATASET_LABELS = {
    "train_subset": r"\texttt{train\_subset}",
    "enron": r"\texttt{enron}",
    "fraudulent_email_corpus": r"\texttt{fraudulent}",
    "spam_ham": r"\texttt{spam\_ham}",
}


def fnum(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def step_sort_value(step: str) -> float:
    if step == "final":
        return math.inf
    return fnum(step)


def pct(value: float) -> float:
    if math.isnan(value):
        return math.nan
    return 100.0 * value


def fmt(value: float, decimals: int = 1) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.{decimals}f}"


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def config_label(index: int) -> str:
    return CONFIG_LABELS.get(index, f"K{index:02d}")


def load_rows() -> list[dict[str, Any]]:
    with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader if row.get("status") == "completed"]
    for row in rows:
        for key in [
            "config_index",
            "rows",
            "ham_count",
            "spam_count",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "spam_prediction_rate",
            "eval_batch_size",
            "max_seq_length",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "runtime_seconds",
        ]:
            row[key] = fnum(row.get(key))
        row["config_index"] = int(row["config_index"])
    return rows


def checkpoint_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["config_index"],
        row["config_id"],
        row["checkpoint_type"],
        row["checkpoint_step"],
        row["checkpoint_path"],
    )


def build_checkpoint_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[checkpoint_key(row)][row["dataset"]] = row

    records: list[dict[str, Any]] = []
    for key, by_dataset in grouped.items():
        config_index, config_id, checkpoint_type, checkpoint_step, checkpoint_path = key
        train = by_dataset.get("train_subset", {})
        enron = by_dataset.get("enron", {})
        fraud = by_dataset.get("fraudulent_email_corpus", {})
        spam_ham = by_dataset.get("spam_ham", {})
        enron_specificity = enron.get("specificity", math.nan)
        fraud_recall = fraud.get("recall", math.nan)
        spam_ham_f1 = spam_ham.get("f1", math.nan)
        external_parts = [enron_specificity, fraud_recall, spam_ham_f1]
        external_score = sum(external_parts) / len(external_parts)
        train_f1 = train.get("f1", math.nan)
        records.append(
            {
                "config_index": config_index,
                "config_label": config_label(config_index),
                "config_id": config_id,
                "checkpoint_type": checkpoint_type,
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": checkpoint_path,
                "step_sort": step_sort_value(checkpoint_step),
                "train_f1": train_f1,
                "enron_specificity": enron_specificity,
                "fraud_recall": fraud_recall,
                "spam_ham_f1": spam_ham_f1,
                "spam_ham_balanced_accuracy": spam_ham.get("balanced_accuracy", math.nan),
                "external_score": external_score,
                "generalization_gap": train_f1 - external_score,
                "max_seq_length": train.get("max_seq_length", math.nan),
                "learning_rate": train.get("learning_rate", math.nan),
                "lora_r": train.get("lora_r", math.nan),
                "lora_alpha": train.get("lora_alpha", math.nan),
                "lora_dropout": train.get("lora_dropout", math.nan),
                "datasets": by_dataset,
            }
        )
    records.sort(key=lambda item: (item["config_index"], item["step_sort"], item["checkpoint_type"]))
    assign_evaluated_checkpoint_numbers(records)
    return records


def assign_evaluated_checkpoint_numbers(records: list[dict[str, Any]]) -> None:
    by_config: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_config[record["config_index"]].append(record)
    for items in by_config.values():
        items.sort(key=lambda item: (item["step_sort"], item["checkpoint_type"]))
        for number, item in enumerate(items, start=1):
            item["evaluated_checkpoint_number"] = number


def best_per_config(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["config_index"]].append(record)
    best = []
    for config_index, items in grouped.items():
        best.append(
            max(
                items,
                key=lambda item: (
                    item["external_score"],
                    item["spam_ham_f1"],
                    -abs(item["generalization_gap"]),
                    item["step_sort"],
                ),
            )
        )
    best.sort(key=lambda item: item["external_score"], reverse=True)
    return best


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def ybar_plot(
    path: Path,
    title_axis: str,
    symbolic_coords: list[str],
    plots: list[tuple[str, str, list[tuple[str, float]]]],
    ymax: int = 105,
    ymin: int = 0,
    ytick: str | None = None,
) -> None:
    coords = ",".join(symbolic_coords)
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.88\textwidth,",
        r"    height=0.46\textwidth,",
        r"    ybar,",
        rf"    ymin={ymin},",
        rf"    ymax={ymax},",
        *( [rf"    ytick={{{ytick}}},"] if ytick else [] ),
        f"    ylabel={{{title_axis}}},",
        f"    symbolic x coords={{{coords}}},",
        r"    xtick=data,",
        r"    xticklabel style={font=\footnotesize, rotate=35, anchor=east},",
        r"    yticklabel style={font=\footnotesize},",
        r"    label style={font=\small},",
        r"    xmajorgrids=false,",
        r"    ymajorgrids,",
        r"    grid style={plotGrid},",
        r"    tick style={draw=none},",
        r"    bar width=5pt,",
        r"    enlarge x limits=0.08,",
        r"    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=2, draw=none, font=\footnotesize},",
        r"]",
    ]
    for legend, color, values in plots:
        lines.extend(
            [
                rf"\addplot[fill={color}, draw=none] coordinates {{",
                *[f"        ({label},{fmt(value)})" for label, value in values],
                r"    };",
            ]
        )
    lines.append(r"\legend{" + ",".join(legend for legend, _, _ in plots) + r"}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    write(path, "\n".join(lines))


def xbar_plot(
    path: Path,
    xlabel: str,
    ordered: list[dict[str, Any]],
    value_key: str,
    color: str,
    xmax: int = 105,
    xmin: int = 0,
    xtick: str | None = None,
) -> None:
    labels = [item["config_label"] for item in ordered]
    ycoords = ",".join(labels)
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.78\textwidth,",
        r"    height=0.52\textwidth,",
        r"    xbar,",
        f"    xmin={xmin},",
        f"    xmax={xmax},",
        *( [f"    xtick={{{xtick}}},"] if xtick else [] ),
        f"    xlabel={{{xlabel}}},",
        f"    symbolic y coords={{{ycoords}}},",
        r"    ytick=data,",
        r"    y dir=reverse,",
        r"    enlarge y limits=0.08,",
        r"    axis x line*=bottom,",
        r"    axis y line*=left,",
        r"    xmajorgrids,",
        r"    grid style={plotGrid},",
        r"    tick style={draw=none},",
        r"    yticklabel style={font=\footnotesize},",
        r"    xticklabel style={font=\footnotesize},",
        r"    label style={font=\small},",
        r"    nodes near coords={\pgfmathprintnumber[fixed, precision=1]{\pgfplotspointmeta}},",
        r"    every node near coord/.append style={font=\scriptsize},",
        r"]",
        rf"\addplot[fill={color}, draw=none, bar width=6pt, point meta=x] coordinates {{",
        *[f"        ({fmt(pct(item[value_key]))},{item['config_label']})" for item in ordered],
        r"    };",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ]
    write(path, "\n".join(lines))


def write_ranking_outputs(best: list[dict[str, Any]]) -> None:
    with RANKING_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rank",
            "config_label",
            "config_index",
            "config_id",
            "checkpoint_step",
            "evaluated_checkpoint_number",
            "external_score",
            "train_f1",
            "enron_specificity",
            "fraud_recall",
            "spam_ham_f1",
            "generalization_gap",
            "max_seq_length",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(best, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **{key: item[key] for key in fieldnames if key in item},
                }
            )

    rows = [
        r"\begin{tabular}{clrrrrr}",
        r"\hline",
        r"\textbf{Miejsce} & \textbf{Konf.} & \textbf{Nr punktu kontrolnego} & \textbf{Wynik zewn. [\%]} & \textbf{\texttt{train\_subset} F1 [\%]} & \textbf{\texttt{spam\_ham} F1 [\%]} & \textbf{Luka [p.p.]} \\",
        r"\hline",
    ]
    for rank, item in enumerate(best[:10], start=1):
        rows.append(
            f"{rank} & \\texttt{{{item['config_label']}}} & {tex_escape(str(item['evaluated_checkpoint_number']))} & "
            f"{fmt(pct(item['external_score']))} & {fmt(pct(item['train_f1']))} & "
            f"{fmt(pct(item['spam_ham_f1']))} & {fmt(pct(item['generalization_gap']))} \\\\"
        )
    rows.extend([r"\hline", r"\end{tabular}"])
    write(TABLES_DIR / "checkpoint_config_ranking.tex", "\n".join(rows))

    mapping_rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"\textbf{Konf.} & \textbf{Kontekst} & \textbf{LR} & \textbf{LoRA r} & \textbf{LoRA alpha} & \textbf{Dropout} \\",
        r"\hline",
    ]
    for item in sorted(best, key=lambda row: row["config_index"]):
        mapping_rows.append(
            f"\\texttt{{{item['config_label']}}} & {int(item['max_seq_length'])} & "
            f"{item['learning_rate']:.0e} & {int(item['lora_r'])} & "
            f"{int(item['lora_alpha'])} & {item['lora_dropout']:.2f} \\\\"
        )
    mapping_rows.extend([r"\hline", r"\end{tabular}"])
    write(TABLES_DIR / "checkpoint_config_mapping.tex", "\n".join(mapping_rows))


def load_rows_from_path(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader if row.get("status") == "completed"]
    for row in rows:
        for key in [
            "config_index",
            "rows",
            "ham_count",
            "spam_count",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "spam_prediction_rate",
            "eval_batch_size",
            "max_seq_length",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "runtime_seconds",
        ]:
            row[key] = fnum(row.get(key))
        row["config_index"] = int(row["config_index"])
    return rows


def build_method_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[checkpoint_key(row)][row["dataset"]] = row

    records: list[dict[str, Any]] = []
    for key, by_dataset in grouped.items():
        config_index, config_id, checkpoint_type, checkpoint_step, checkpoint_path = key
        if not all(dataset in by_dataset for dataset in DATASET_LABELS):
            continue
        train = by_dataset["train_subset"]
        enron = by_dataset["enron"]
        fraud = by_dataset["fraudulent_email_corpus"]
        spam_ham = by_dataset["spam_ham"]
        external_score = (enron["specificity"] + fraud["recall"] + spam_ham["f1"]) / 3
        records.append(
            {
                "config_index": config_index,
                "config_label": f"K{config_index:02d}",
                "config_id": config_id,
                "checkpoint_type": checkpoint_type,
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": checkpoint_path,
                "step_sort": step_sort_value(checkpoint_step),
                "train_f1": train["f1"],
                "enron_specificity": enron["specificity"],
                "fraud_recall": fraud["recall"],
                "spam_ham_f1": spam_ham["f1"],
                "external_score": external_score,
                "generalization_gap": train["f1"] - external_score,
                "max_seq_length": train["max_seq_length"],
                "learning_rate": train["learning_rate"],
                "lora_r": train["lora_r"],
                "lora_alpha": train["lora_alpha"],
                "lora_dropout": train["lora_dropout"],
                "datasets": by_dataset,
            }
        )
    records.sort(key=lambda item: (item["config_index"], item["step_sort"], item["checkpoint_type"]))
    assign_evaluated_checkpoint_numbers(records)
    return records


def best_method_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        records,
        key=lambda item: (
            item["external_score"],
            item["spam_ham_f1"],
            -abs(item["generalization_gap"]),
            item["step_sort"],
        ),
    )


def write_method_comparison_outputs() -> list[dict[str, Any]]:
    best_records: list[dict[str, Any]] = []
    records_by_method: dict[str, list[dict[str, Any]]] = {}
    for method_id, path in METHOD_SUMMARY_PATHS.items():
        method_rows = load_rows_from_path(path)
        records = build_method_records(method_rows)
        records_by_method[method_id] = records
        best = best_method_record(records)
        best["method_id"] = method_id
        best["method_label"] = METHOD_LABELS[method_id]
        best["method_name"] = METHOD_NAMES[method_id]
        best["evaluated_configs_count"] = len({record["config_index"] for record in records})
        best["evaluated_checkpoints_count"] = len(records)
        best["selected_config_checkpoints_count"] = len(
            [record for record in records if record["config_index"] == best["config_index"]]
        )
        best_records.append(best)

    best_records.sort(key=lambda item: item["method_id"])

    rows = [
        r"\begin{tabular}{llcrrrrr}",
        r"\hline",
        r"\textbf{Metoda} & \textbf{Konf.} & \textbf{Nr punktu} & \textbf{\texttt{train\_subset} F1 [\%]} & \textbf{\texttt{enron} spec. [\%]} & \textbf{\texttt{fraudulent} recall [\%]} & \textbf{\texttt{spam\_ham} F1 [\%]} & \textbf{\(S_{\mathrm{ext}}\) [\%]} \\",
        r"\hline",
    ]
    for item in best_records:
        method_label = rf"\texttt{{{item['method_label']}}}"
        if item["method_id"] == "04":
            method_label += r"$^{*}$"
        rows.append(
            f"{method_label} & \\texttt{{{item['config_label']}}} & {tex_escape(str(item['evaluated_checkpoint_number']))} & "
            f"{fmt(pct(item['train_f1']))} & {fmt(pct(item['enron_specificity']))} & "
            f"{fmt(pct(item['fraud_recall']))} & {fmt(pct(item['spam_ham_f1']))} & "
            f"{fmt(pct(item['external_score']))} \\\\"
        )
    rows.extend([r"\hline", r"\end{tabular}"])
    write(TABLES_DIR / "method_comparison_best.tex", "\n".join(rows))

    labels = [item["method_label"] for item in best_records]
    ybar_plot(
        PLOTS_DIR / "method_comparison_dataset_metrics.tex",
        "Wartość metryki [\\%]",
        labels,
        [
            ("train\\_subset F1", "plotBlue", [(item["method_label"], pct(item["train_f1"])) for item in best_records]),
            ("enron specificity", "plotHam", [(item["method_label"], pct(item["enron_specificity"])) for item in best_records]),
            ("fraud recall", "plotSpam", [(item["method_label"], pct(item["fraud_recall"])) for item in best_records]),
            ("spam\\_ham F1", "plotAccent", [(item["method_label"], pct(item["spam_ham_f1"])) for item in best_records]),
        ],
        ymax=100,
        ymin=93,
        ytick="93,94,95,96,97,98,99,100",
    )
    write_method_error_tradeoff(best_records)
    write_method_training_outputs(best_records, records_by_method)
    return best_records


def write_method_error_tradeoff(best_records: list[dict[str, Any]]) -> None:
    colors = {
        "01": "plotBlue",
        "02": "plotHam",
        "03": "plotAccent",
        "04": "plotSpam",
    }
    marks = {
        "01": "*",
        "02": "square*",
        "03": "triangle*",
        "04": "diamond*",
    }
    max_x = max(pct(1.0 - item["enron_specificity"]) for item in best_records)
    max_y = max(pct(1.0 - item["fraud_recall"]) for item in best_records)
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.78\textwidth,",
        r"    height=0.48\textwidth,",
        r"    xmin=0,",
        rf"    xmax={math.ceil(max_x) + 1},",
        r"    ymin=0,",
        rf"    ymax={max(2, math.ceil(max_y) + 1)},",
        r"    xlabel={False positive rate na \texttt{enron} [\%]},",
        r"    ylabel={False negative rate na \texttt{fraudulent} [\%]},",
        r"    xmajorgrids,",
        r"    ymajorgrids,",
        r"    grid style={plotGrid},",
        r"    tick style={draw=none},",
        r"    xticklabel style={font=\footnotesize},",
        r"    yticklabel style={font=\footnotesize},",
        r"    label style={font=\small},",
        r"    legend style={at={(0.5,-0.22)}, anchor=north, legend columns=4, draw=none, font=\footnotesize},",
        r"]",
    ]
    legends = []
    for item in best_records:
        method_id = item["method_id"]
        x = pct(1.0 - item["enron_specificity"])
        y = pct(1.0 - item["fraud_recall"])
        label = item["method_label"]
        lines.append(
            rf"\addplot+[only marks, mark={marks[method_id]}, mark size=2.2pt, color={colors[method_id]}] "
            rf"coordinates {{({fmt(x, 2)},{fmt(y, 2)})}};"
        )
        lines.append(rf"\node[font=\scriptsize, anchor=west] at (axis cs:{fmt(x, 2)},{fmt(y, 2)}) {{\texttt{{{label}}}}};")
        legends.append(rf"\texttt{{{label}}}")
    lines.append(r"\legend{" + ",".join(legends) + r"}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    write(PLOTS_DIR / "method_error_tradeoff.tex", "\n".join(lines))


def write_method_training_outputs(
    best_records: list[dict[str, Any]],
    records_by_method: dict[str, list[dict[str, Any]]],
) -> None:
    rows = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"\textbf{Metoda} & \textbf{Konf.} & \textbf{Najlepszy nr punktu} & \textbf{Krok} & \textbf{\(S_{\mathrm{ext}}\) [\%]} & \textbf{Punkty konf.} \\",
        r"\hline",
    ]

    by_method_best = {item["method_id"]: item for item in best_records}
    for method_id in sorted(by_method_best):
        item = by_method_best[method_id]
        method_label = rf"\texttt{{{item['method_label']}}}"
        if item["method_id"] == "04":
            method_label += r"$^{*}$"
        rows.append(
            f"{method_label} & \\texttt{{{item['config_label']}}} & "
            f"{tex_escape(str(item['evaluated_checkpoint_number']))} & "
            f"{tex_escape(str(item['checkpoint_step']))} & "
            f"{fmt(pct(item['external_score']))} & "
            f"{item['selected_config_checkpoints_count']} \\\\"
        )
    rows.extend([r"\hline", r"\end{tabular}"])
    write(TABLES_DIR / "method_training_convergence.tex", "\n".join(rows))

    def small_axis(method_id: str) -> str:
        best = by_method_best[method_id]
        method_records = [
            record
            for record in records_by_method[method_id]
            if record["config_index"] == best["config_index"]
        ]
        method_records.sort(key=lambda item: item["evaluated_checkpoint_number"])
        external_coords = " ".join(
            f"({item['evaluated_checkpoint_number']},{fmt(pct(item['external_score']))})"
            for item in method_records
        )
        train_coords = " ".join(
            f"({item['evaluated_checkpoint_number']},{fmt(pct(item['train_f1']))})"
            for item in method_records
        )
        best_x = best["evaluated_checkpoint_number"]
        title = rf"\texttt{{{METHOD_LABELS[method_id]}}}: \texttt{{{best['config_label']}}}"
        return "\n".join(
            [
                r"\begin{tikzpicture}",
                r"\begin{axis}[",
                r"    width=0.455\textwidth,",
                r"    height=0.285\textwidth,",
                rf"    title={{{title}}},",
                r"    title style={font=\scriptsize, yshift=-1.2ex},",
                r"    xmin=1,",
                r"    xmax=10,",
                r"    ymin=75,",
                r"    ymax=101,",
                r"    xtick={1,2,3,4,5,6,7,8,9,10},",
                r"    ytick={75,80,85,90,95,100},",
                r"    xlabel={Nr punktu kontrolnego},",
                r"    ylabel={Wynik [\%]},",
                r"    xmajorgrids,",
                r"    ymajorgrids,",
                r"    grid style={plotGrid},",
                r"    tick style={draw=none},",
                r"    xticklabel style={font=\tiny},",
                r"    yticklabel style={font=\tiny},",
                r"    label style={font=\scriptsize},",
                r"]",
                rf"\addplot+[mark=*, mark size=1.1pt, line width=0.75pt, color=plotBlue] coordinates {{{train_coords}}};",
                rf"\addplot+[mark=square*, mark size=1.1pt, line width=0.75pt, color=plotAccent] coordinates {{{external_coords}}};",
                rf"\addplot[densely dashed, color=black!45] coordinates {{({best_x},75) ({best_x},101)}};",
                r"\end{axis}",
                r"\end{tikzpicture}",
            ]
        )

    legend = (
        r"{\scriptsize "
        r"\tikz[baseline=-0.5ex]\draw[plotBlue, line width=0.8pt] (0,0) -- (0.45,0);~F1 na \texttt{train\_subset}"
        r"\quad "
        r"\tikz[baseline=-0.5ex]\draw[plotAccent, line width=0.8pt] (0,0) -- (0.45,0);~\(S_{\mathrm{ext}}\)"
        r"\quad "
        r"\tikz[baseline=-0.5ex]\draw[black!45, densely dashed, line width=0.8pt] (0,0) -- (0.45,0);~najlepszy punkt"
        r"}\\[0.3em]"
    )
    method_ids = sorted(by_method_best)
    lines = [
        r"\begingroup",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.84}",
        legend,
        r"\begin{tabular}{cc}",
        small_axis(method_ids[0]) + " & " + small_axis(method_ids[1]) + r" \\",
        small_axis(method_ids[2]) + " & " + small_axis(method_ids[3]) + r" \\",
        r"\end{tabular}",
        r"\endgroup",
    ]
    write(PLOTS_DIR / "method_training_trajectories.tex", "\n".join(lines))


def write_dataset_metric_bars(best: list[dict[str, Any]]) -> None:
    top = sorted(best[:5], key=lambda item: item["config_index"])
    labels = [item["config_label"] for item in top]
    ybar_plot(
        PLOTS_DIR / "checkpoint_top_config_dataset_metrics.tex",
        "Wartość metryki [\\%]",
        labels,
        [
            ("train\\_subset F1", "plotBlue", [(item["config_label"], pct(item["train_f1"])) for item in top]),
            ("enron specificity", "plotHam", [(item["config_label"], pct(item["enron_specificity"])) for item in top]),
            ("fraud recall", "plotSpam", [(item["config_label"], pct(item["fraud_recall"])) for item in top]),
            ("spam\\_ham F1", "plotAccent", [(item["config_label"], pct(item["spam_ham_f1"])) for item in top]),
        ],
        ymax=100,
        ymin=95,
        ytick="95,96,97,98,99,100",
    )


def write_generalization_gap(best: list[dict[str, Any]]) -> None:
    ordered = sorted(best, key=lambda item: item["generalization_gap"])
    xbar_plot(
        PLOTS_DIR / "checkpoint_generalization_gap.tex",
        "Różnica: F1 na train\\_subset minus wynik zewnętrzny [p.p.]",
        ordered,
        "generalization_gap",
        "plotAccent",
        xmax=35,
    )


def write_external_score_ranking(best: list[dict[str, Any]]) -> None:
    ordered = sorted(best, key=lambda item: item["external_score"])
    xbar_plot(
        PLOTS_DIR / "checkpoint_external_score_ranking.tex",
        "Najlepszy wynik zewnętrzny [\\%]",
        ordered,
        "external_score",
        "plotBlue",
        xmax=100,
        xmin=84,
        xtick="84,88,92,96,100",
    )


def write_trajectory(records: list[dict[str, Any]], best: list[dict[str, Any]]) -> None:
    top_indices = {item["config_index"] for item in best[:5]}
    by_config: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["config_index"] in top_indices:
            by_config[record["config_index"]].append(record)
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.88\textwidth,",
        r"    height=0.46\textwidth,",
        r"    xmin=1,",
        r"    xmax=10,",
        r"    ymin=60,",
        r"    ymax=105,",
        r"    xlabel={Nr punktu kontrolnego},",
        r"    ylabel={Wynik zewnętrzny [\%]},",
        r"    xmajorgrids,",
        r"    ymajorgrids,",
        r"    grid style={plotGrid},",
        r"    tick style={draw=none},",
        r"    xtick={1,2,3,4,5,6,7,8,9,10},",
        r"    xticklabel style={font=\footnotesize},",
        r"    yticklabel style={font=\footnotesize},",
        r"    label style={font=\small},",
        r"    legend style={at={(0.5,-0.20)}, anchor=north, legend columns=5, draw=none, font=\footnotesize},",
        r"]",
    ]
    colors = ["plotBlue", "plotHam", "plotSpam", "plotAccent", "black!65"]
    legends = []
    for color, config_index in zip(colors, sorted(top_indices)):
        items = sorted(by_config[config_index], key=lambda item: item["step_sort"])
        coords = " ".join(f"({i},{fmt(pct(item['external_score']))})" for i, item in enumerate(items, start=1))
        lines.append(rf"\addplot+[mark=*, line width=0.8pt, color={color}] coordinates {{{coords}}};")
        legends.append(config_label(config_index))
    lines.append(r"\legend{" + ",".join(rf"\texttt{{{legend}}}" for legend in legends) + r"}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    write(PLOTS_DIR / "checkpoint_external_score_trajectory.tex", "\n".join(lines))


def write_accuracy_grid(records: list[dict[str, Any]]) -> None:
    by_config: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_config[record["config_index"]].append(record)

    dataset_specs = [
        ("train_subset", r"\texttt{train\_subset}", "plotBlue"),
        ("enron", r"\texttt{enron}", "plotHam"),
        ("fraudulent_email_corpus", r"\texttt{fraudulent}", "plotSpam"),
        ("spam_ham", r"\texttt{spam\_ham}", "plotAccent"),
    ]

    def small_axis(config_index: int) -> str:
        items = sorted(by_config[config_index], key=lambda item: item["evaluated_checkpoint_number"])
        axis_lines = [
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"    width=0.242\textwidth,",
            r"    height=0.165\textwidth,",
            rf"    title={{\texttt{{{config_label(config_index)}}}}},",
            r"    title style={font=\scriptsize, yshift=-1.5ex},",
            r"    xmin=1,",
            r"    xmax=10,",
            r"    ymin=50,",
            r"    ymax=102,",
            r"    xtick={1,5,10},",
            r"    ytick={60,80,100},",
            r"    xmajorgrids,",
            r"    ymajorgrids,",
            r"    grid style={plotGrid},",
            r"    tick style={draw=none},",
            r"    xticklabel style={font=\tiny},",
            r"    yticklabel style={font=\tiny},",
            r"    tick align=outside,",
            r"]",
        ]
        for dataset, _, color in dataset_specs:
            coords = []
            for item in items:
                row = item["datasets"].get(dataset)
                if not row:
                    continue
                coords.append(f"({item['evaluated_checkpoint_number']},{fmt(pct(row['accuracy']))})")
            axis_lines.append(rf"\addplot+[mark=none, line width=0.55pt, color={color}] coordinates {{{' '.join(coords)}}};")
        axis_lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
        return "\n".join(axis_lines)

    legend_parts = []
    for _, label, color in dataset_specs:
        legend_parts.append(
            rf"\tikz[baseline=-0.5ex]\draw[{color}, line width=0.8pt] (0,0) -- (0.45,0);~{{\scriptsize {label}}}"
        )

    lines = [
        r"\begingroup",
        r"\centering",
        r"\setlength{\tabcolsep}{1pt}",
        r"\renewcommand{\arraystretch}{0.76}",
        r"{\scriptsize " + r"\quad ".join(legend_parts) + r"}\\[0.25em]",
        r"\begin{tabular}{cccc}",
    ]
    config_indexes = sorted(by_config)
    for row_start in range(0, len(config_indexes), 4):
        row_items = config_indexes[row_start : row_start + 4]
        lines.append((" & ".join(small_axis(config_index) for config_index in row_items)) + r" \\")
    lines.extend([r"\end{tabular}", r"\endgroup"])
    write(PLOTS_DIR / "checkpoint_accuracy_grid.tex", "\n".join(lines))


def write_error_profile(best: list[dict[str, Any]]) -> None:
    item = best[0]
    datasets = ["train_subset", "enron", "fraudulent_email_corpus", "spam_ham"]
    labels = ["train", "enron", "fraud", "spamham"]
    fpr_values = []
    fnr_values = []
    for label, dataset in zip(labels, datasets):
        row = item["datasets"][dataset]
        fpr_values.append((label, pct(row["false_positive_rate"])))
        fnr_values.append((label, pct(row["false_negative_rate"])))
    ybar_plot(
        PLOTS_DIR / "checkpoint_best_error_rates.tex",
        "Odsetek błędów [\\%]",
        labels,
        [
            ("false positive rate", "plotHam", fpr_values),
            ("false negative rate", "plotSpam", fnr_values),
        ],
        ymax=4,
    )


def main() -> int:
    global CONFIG_LABELS
    rows = load_rows()
    CONFIG_LABELS = {
        config_index: f"K{offset:02d}"
        for offset, config_index in enumerate(sorted({row["config_index"] for row in rows}), start=1)
    }
    records = build_checkpoint_records(rows)
    best = best_per_config(records)
    write_ranking_outputs(best)
    write_external_score_ranking(best)
    write_generalization_gap(best)
    write_dataset_metric_bars(best)
    write_trajectory(records, best)
    write_accuracy_grid(records)
    write_error_profile(best)
    method_best = write_method_comparison_outputs()
    print(f"Loaded rows: {len(rows)}")
    print(f"Checkpoint records: {len(records)}")
    print(f"Configs ranked: {len(best)}")
    print(f"Best config: {best[0]['config_label']} {best[0]['config_id']} evaluated_checkpoint_no={best[0]['evaluated_checkpoint_number']} external={pct(best[0]['external_score']):.1f}%")
    print(
        "Best method: "
        f"{max(method_best, key=lambda item: item['external_score'])['method_label']} "
        f"external={pct(max(method_best, key=lambda item: item['external_score'])['external_score']):.1f}%"
    )
    print(f"Wrote plots to: {PLOTS_DIR}")
    print(f"Wrote ranking table to: {TABLES_DIR / 'checkpoint_config_ranking.tex'}")
    print(f"Wrote config mapping table to: {TABLES_DIR / 'checkpoint_config_mapping.tex'}")
    print(f"Wrote ranking CSV to: {RANKING_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
