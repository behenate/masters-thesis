# Inference benchmark results

Date: 2026-07-18

Hardware: Apple M3 Pro, 12 CPU cores, 18 GiB unified memory, Apple MPS,
macOS 26.5.1.

Every method classified the same deterministic sample of 1000 `spam_ham`
messages (716 ham, 284 spam). Sample SHA-256:
`5589812e2fd4a0965ed26c14d32a52cf182859ecd6877517717a23f68abb59c7`.

| Method | Accuracy [%] | F1 [%] | Inference [s] | Messages/s | Mean [ms/message] | Batch | Peak RSS [MiB] | Peak MPS driver [MiB] | Artifacts [MiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + NB | 89.8 | 83.5 | 0.095 | 10558.3 | 0.095 | 1000 | 337.1 | 0.0 | 13.0 |
| TF-IDF + LR | 74.0 | 68.5 | 0.096 | 10461.7 | 0.096 | 1000 | 350.3 | 0.0 | 9.2 |
| TF-IDF + SVM | 91.1 | 85.9 | 0.450 | 2221.3 | 0.450 | 1000 | 370.8 | 0.0 | 8.1 |
| fastText | 72.2 | 66.3 | 0.114 | 8759.1 | 0.114 | 1000 | 1391.5 | 0.0 | 980.3 |
| MiniLM + LR | 73.5 | 66.3 | 2.970 | 336.7 | 2.970 | 64 | 809.1 | 1708.5 | 87.3 |
| DistilBERT | 91.8 | 87.2 | 17.280 | 57.9 | 17.280 | 32 | 563.2 | 3034.8 | 256.1 |
| Qwen3 + LoRA, method 03 | 97.2 | 95.2 | 136.720 | 7.31 | 136.720 | 16 | 3465.9 | 6535.5 | 1525.9 |

Inference time includes preprocessing needed to obtain a prediction, including
TF-IDF vectorization, MiniLM encoding and Qwen tokenization. Model loading and
one warm-up pass are excluded. The mean time per message is the total batch
runtime divided by 1000; it is not a single-request latency percentile.

`Peak RSS` is the maximum resident size of the isolated worker process. On
Apple Silicon, MPS uses unified system memory. Consequently, `Peak RSS` and
`Peak MPS driver` overlap and must not be added together. The latter indicates
memory attributed by the Metal driver rather than dedicated VRAM. Classical
CPU methods therefore report zero MPS memory.

Artifact size covers files required for inference. For MiniLM it includes the
encoder and logistic-regression classifier. For Qwen3 it includes the base
model and LoRA adapter; the adapter alone occupies 77.1 MiB.

Raw values and additional metrics are available in `inference_benchmark.csv`.
Machine metadata is stored in `hardware.json`.
