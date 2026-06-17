# MiniLM embeddings + Logistic Regression

Baseline based on sentence embeddings from a compact MiniLM/Sentence-BERT model
and a logistic-regression classifier.

## Run

Install the Sentence-BERT dependency in the project venv:

```bash
./.venv/bin/python -m pip install sentence-transformers
```

Run the baseline with the thesis defaults:

```bash
./.venv/bin/python sota/minilm_logistic_regression/run_minilm_logistic_regression.py
```

By default the script trains on the `training_all` 92/2/6 training split with
seed `67`, evaluates `train_subset`, `enron`, `fraudulent_email_corpus`, and
`spam_ham` with sample limit `2000`, and writes `summary.csv` in this folder.

## Parameter selection

Training and validation embeddings are cached by `sota/tune_baselines.py`. The
search compared 256- and 512-token truncation, mean pooling over 256-token
chunks, and separate subject/body embeddings. Each representation was tested
with normalized and raw embeddings, `C` values `0.01`, `0.1`, `1`, and `10`,
and optional balanced class weights. The classification threshold was selected
on validation predictions.

Configurations were ranked by mean validation F1 calculated separately for
the source corpora in `training_all`, with overall validation F1 used as a
tie-breaker. The selected frozen-encoder model uses separately encoded subject
and body vectors, L2 normalization, `C=10`, no class weighting, and threshold
`0.5808627`. Fine-tuning the encoder was intentionally excluded because it
would constitute a different baseline. External datasets did not affect the
selection.
