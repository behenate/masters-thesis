#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOTA = ROOT / "sota"
THESIS = ROOT / "thesis" / "tex"
QWEN_SUMMARY = (
    ROOT / "lora-fine-tuning" / "methods" / "03_causal_lm_next_token" / "notebooks"
    / "checkpoint_eval_20260605_151634" / "summary.csv"
)

METHODS = [
    ("tfidf_naive_bayes", "TF-IDF + NB"),
    ("tfidf_logistic_regression", "TF-IDF + LR"),
    ("tfidf_linear_svm", "TF-IDF + SVM"),
    ("fasttext", r"\texttt{fastText}"),
    ("minilm_logistic_regression", "MiniLM + LR"),
    ("distilbert_sequence_classification", r"\texttt{DistilBERT}"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_by_dataset(path: Path) -> dict[str, dict[str, str]]:
    rows = [row for row in read_csv(path) if row.get("status", "completed") == "completed"]
    result = {row["dataset"]: row for row in rows}
    missing = {"train_subset", "enron", "fraudulent_email_corpus", "spam_ham"} - result.keys()
    if missing:
        raise RuntimeError(f"Missing completed rows in {path}: {sorted(missing)}")
    return result


def metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def quality_values(rows: dict[str, dict[str, str]]) -> dict[str, float]:
    values = {
        "train": metric(rows["train_subset"], "f1"),
        "enron": metric(rows["enron"], "specificity"),
        "fraud": metric(rows["fraudulent_email_corpus"], "recall"),
        "spam_ham": metric(rows["spam_ham"], "f1"),
    }
    values["external"] = (values["enron"] + values["fraud"] + values["spam_ham"]) / 3
    return values


def qwen_values() -> dict[str, float]:
    selected = {}
    for row in read_csv(QWEN_SUMMARY):
        if row["config_index"] == "8" and row["checkpoint_step"] == "3000" and row["status"] == "completed":
            selected[row["dataset"]] = row
    return quality_values(selected)


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def write_quality_table(values: dict[str, dict[str, float]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"\textbf{Metoda} & \textbf{Train F1 [\%]} & \textbf{Enron spec. [\%]} & \textbf{Fraud recall [\%]} & \textbf{SpamHam F1 [\%]} & \textbf{\(S_{\mathrm{ext}}\) [\%]} \\",
        r"\hline",
    ]
    for method, label in METHODS:
        item = values[method]
        lines.append(
            f"{label} & {pct(item['train'])} & {pct(item['enron'])} & {pct(item['fraud'])} & "
            f"{pct(item['spam_ham'])} & {pct(item['external'])} \\\\"
        )
    item = values["qwen3_lora_next_token"]
    lines.extend([
        f"Qwen3 + LoRA, \\texttt{{03}} & {pct(item['train'])} & {pct(item['enron'])} & {pct(item['fraud'])} & {pct(item['spam_ham'])} & {pct(item['external'])} \\\\",
        r"\hline",
        r"\end{tabular}",
    ])
    (THESIS / "tables" / "sota_baseline_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_configuration_table() -> None:
    configs = {
        method: json.loads((SOTA / method / "tuned_config.json").read_text(encoding="utf-8"))
        for method in ("tfidf_logistic_regression", "tfidf_linear_svm", "fasttext", "minilm_logistic_regression")
    }
    distil = json.loads(
        (SOTA / "distilbert_sequence_classification" / "trained_model" / "evaluation_config.json").read_text(encoding="utf-8")
    )
    rows = [
        ("TF-IDF + LR", "słowa 1--2", r"\(C=10\)", configs["tfidf_logistic_regression"]["threshold"], configs["tfidf_logistic_regression"]["validation_metrics"]["f1"]),
        ("TF-IDF + SVM", "znaki 3--5", r"\(C=10\)", configs["tfidf_linear_svm"]["threshold"], configs["tfidf_linear_svm"]["validation_metrics"]["f1"]),
        (r"\texttt{fastText}", "znaki 3--6, słowa 1", r"\(lr=0{,}25\), 20 epok, \(d=100\)", configs["fasttext"]["threshold"], configs["fasttext"]["validation_metrics"]["f1"]),
        ("MiniLM + LR", "temat i treść osobno", r"po 256 tokenów, norm., \(C=10\)", configs["minilm_logistic_regression"]["threshold"], configs["minilm_logistic_regression"]["validation_metrics"]["f1"]),
        (r"\texttt{DistilBERT}", "klasyfikacja sekwencji", r"1 epoka, 512 tokenów, \(lr=2\cdot10^{-5}\)", distil["decision_threshold"], distil["validation_metrics"]["f1"]),
    ]
    lines = [
        r"\begin{tabular}{llllr}", r"\hline",
        r"\textbf{Metoda} & \textbf{Reprezentacja} & \textbf{Parametry} & \textbf{Próg} & \textbf{Wal. F1 [\%]} \\",
        r"\hline",
    ]
    for method, representation, parameters, threshold, validation_f1 in rows:
        lines.append(f"{method} & {representation} & {parameters} & {threshold:.3f} & {validation_f1 * 100:.1f} \\\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (THESIS / "tables" / "sota_tuned_configurations.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_plot(values: dict[str, dict[str, float]]) -> None:
    coords = {
        "NB": values["tfidf_naive_bayes"]["external"] * 100,
        "LR": values["tfidf_logistic_regression"]["external"] * 100,
        "SVM": values["tfidf_linear_svm"]["external"] * 100,
        "fastText": values["fasttext"]["external"] * 100,
        "MiniLM": values["minilm_logistic_regression"]["external"] * 100,
        "DistilBERT": values["distilbert_sequence_classification"]["external"] * 100,
        "Qwen": values["qwen3_lora_next_token"]["external"] * 100,
    }
    plot = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.90\textwidth, height=0.49\textwidth, ybar,
    ymin=75, ymax=100, ytick={75,80,85,90,95,100}, ylabel={\(S_{\mathrm{ext}}\) [\%]},
    symbolic x coords={fastText,LR,MiniLM,SVM,NB,DistilBERT,Qwen},
    xtick={fastText,LR,MiniLM,SVM,NB,DistilBERT,Qwen},
    xticklabel style={font=\footnotesize, rotate=35, anchor=east},
    yticklabel style={font=\footnotesize}, label style={font=\small},
    ymajorgrids, grid style={plotGrid}, tick style={draw=none}, bar width=12pt,
    enlarge x limits=0.10,
    nodes near coords={\pgfmathprintnumber[fixed, precision=1]{\pgfplotspointmeta}},
    every node near coord/.append style={font=\scriptsize},
    legend style={at={(0.5,-0.28)}, anchor=north, legend columns=2, draw=none, font=\footnotesize},
]
\addplot[fill=plotBlue, draw=black, line width=0.2pt, point meta=y] coordinates {
COORDS_BASE
};
\addplot[fill=plotSpam, draw=black, line width=0.2pt, point meta=y] coordinates {(Qwen,QWEN)};
\legend{Metody bazowe,Qwen3-0.6B + LoRA}
\end{axis}
\end{tikzpicture}
"""
    base_order = ("fastText", "LR", "MiniLM", "SVM", "NB", "DistilBERT")
    base = "\n".join(f"        ({name},{coords[name]:.1f})" for name in base_order)
    plot = plot.replace("COORDS_BASE", base).replace("QWEN", f"{coords['Qwen']:.1f}")
    (THESIS / "plots" / "sota_baseline_external_score.tex").write_text(plot, encoding="utf-8")


def write_inference_outputs() -> None:
    benchmark = {row["method"]: row for row in read_csv(SOTA / "deployment_benchmark" / "inference_benchmark.csv")}
    order = [method for method, _ in METHODS] + ["qwen3_lora_next_token"]
    labels = dict(METHODS + [("qwen3_lora_next_token", r"Qwen3 + LoRA, \texttt{03}")])
    lines = [
        r"\begin{tabular}{lrrrrrr}", r"\hline",
        r"\textbf{Metoda} & \textbf{Partia} & \textbf{Ładowanie [s]} & \textbf{Inferencja [s]} & \textbf{Wiad./s} & \textbf{RAM [MiB]} & \textbf{Model [MiB]} \\",
        r"\hline",
    ]
    for method in order:
        row = benchmark[method]
        lines.append(
            f"{labels[method]} & {int(float(row['batch_size']))} & {float(row['load_seconds']):.3f} & "
            f"{float(row['inference_seconds']):.3f} & {float(row['throughput_samples_per_second']):.1f} & "
            f"{float(row['peak_rss_mib']):.1f} & {float(row['artifact_size_mib']):.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    (THESIS / "tables" / "sota_inference_costs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_labels = {
        "tfidf_naive_bayes": "NB", "tfidf_logistic_regression": "LR", "tfidf_linear_svm": "SVM",
        "fasttext": "fastText", "minilm_logistic_regression": "MiniLM",
        "distilbert_sequence_classification": "DistilBERT", "qwen3_lora_next_token": "Qwen",
    }
    coordinates = "\n".join(
        f"    ({float(benchmark[method]['throughput_samples_per_second']):.1f},{plot_labels[method]})"
        for method in order
    )
    plot = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.90\textwidth, height=0.54\textwidth, xbar, xmode=log, log basis x=10,
    xmin=5, xmax=20000, xlabel={Wiadomości na sekundę},
    symbolic y coords={Qwen,DistilBERT,MiniLM,fastText,SVM,LR,NB},
    ytick={Qwen,DistilBERT,MiniLM,fastText,SVM,LR,NB},
    yticklabels={Qwen3 + LoRA,DistilBERT,MiniLM + LR,fastText,TF-IDF + SVM,TF-IDF + LR,TF-IDF + NB},
    xtick={10,100,1000,10000}, xticklabels={$10$,$100$,$1000$,$10000$},
    xmajorgrids, grid style={plotGrid}, tick style={draw=none}, bar width=8pt,
    yticklabel style={font=\footnotesize}, xticklabel style={font=\footnotesize}, label style={font=\small},
]
\addplot[fill=plotBlue!72, draw=plotBlue!85!black, line width=0.25pt] coordinates {
COORDINATES
};
\end{axis}
\end{tikzpicture}
""".replace("COORDINATES", coordinates)
    (THESIS / "plots" / "sota_inference_throughput.tex").write_text(plot, encoding="utf-8")


def main() -> int:
    values = {
        method: quality_values(completed_by_dataset(SOTA / method / "summary.csv"))
        for method, _ in METHODS
    }
    values["qwen3_lora_next_token"] = qwen_values()
    write_quality_table(values)
    write_configuration_table()
    write_quality_plot(values)
    write_inference_outputs()
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
