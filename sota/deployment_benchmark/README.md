# Deployment inference benchmark

The benchmark compares the persisted SOTA baselines and the selected Qwen3 LoRA
checkpoint on the same deterministic, stratified sample of 1000 `spam_ham`
messages. Every method runs in a fresh process so that peak memory values do not
include models left behind by an earlier test.

Reported values include model loading time, inference time, throughput, mean
time per message, process RSS, accelerator memory and the size of deployment
artifacts. Inference time includes all preprocessing needed by the method, such
as TF-IDF vectorization or tokenization, but excludes model loading and warm-up.

Prepare the persisted artifacts of the lightweight baselines:

```bash
for method in \
  tfidf_naive_bayes \
  tfidf_logistic_regression \
  tfidf_linear_svm \
  fasttext \
  minilm_logistic_regression
do
  ./.venv/bin/python sota/deployment_benchmark/prepare_artifact.py \
    --method "$method"
done
```

Train DistilBERT separately before running its deployment benchmark:

```bash
./.venv/bin/python \\
  sota/distilbert_sequence_classification/run_distilbert_sequence_classification.py
```

Run all deployment tests:

```bash
./.venv/bin/python sota/deployment_benchmark/run_benchmark.py --overwrite
```

The combined results are written to `inference_benchmark.csv`; machine details
are stored in `hardware.json`. Generated models, samples and intermediate worker
results are ignored by Git.

On Apple Silicon, MPS and the CPU share unified memory. The reported process RSS
and MPS driver allocation therefore overlap and must not be added together.
