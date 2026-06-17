# DistilBERT sequence classification

Baseline using a Hugging Face sequence-classification model based on DistilBERT.

Run from the repository root:

```bash
./.venv/bin/python \\
  sota/distilbert_sequence_classification/run_distilbert_sequence_classification.py
```

The script fine-tunes `distilbert-base-uncased` for one epoch on the 92% training
split, using seed `67`, a 512-token context, training batch size `16`, learning
rate `2e-5`, weight decay `0.01`, and a linear schedule with 6% warm-up. The
spam threshold is selected on the 2% validation split. External datasets are
evaluated only after this selection.

The completed run used threshold `0.462413`, reached validation F1 `0.995506`,
and took `7650.4` seconds for the training epoch.

The final metrics are written to `summary.csv`. The trained model, tokenizer,
selected threshold, and validation metadata are saved in `trained_model/`;
this directory is intentionally ignored by Git because it contains the full
model weights.
