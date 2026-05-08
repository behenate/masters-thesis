# Qwen3 0.6B Method Layout

This directory groups the spam-classification approaches in a thesis-friendly order. Duplicate legacy notebook copies were removed from the old locations, while shared scripts and existing result folders were left in place for compatibility.

## 01 Sequence Classification

Uses a classification head (`AutoModelForSequenceClassification`) instead of prompting the model to produce label text.

- Notebook: `01_sequence_classification/notebooks/qwen3_0.6b_sequence_classification.ipynb`
- New results should go under `01_sequence_classification/results/`

## 02 Causal LM: Generation + Parsing

Trains a causal LM to output `ham` or `spam`, then evaluates by generating a short continuation and parsing the generated text. This is useful as the thesis stepping-stone because it exposes the fragility of text parsing and parse failures.

- Historical notebook: `02_causal_lm_generation_parsing/notebooks/qwen3_0.6b_casual_lm_generation_parsing.ipynb`
- Sweep launcher notebook: `02_causal_lm_generation_parsing/notebooks/qwen3_0.6b_generation_parsing_sweep.ipynb`
- Sweep script: `02_causal_lm_generation_parsing/qwen3_0.6b_generation_parsing_sweep.py`
- New results: `02_causal_lm_generation_parsing/results/`

Example smoke run:

```bash
python lora-fine-tuning/methods/02_causal_lm_generation_parsing/qwen3_0.6b_generation_parsing_sweep.py run-sweep \
  --config-index 1 \
  --allow-non-cuda \
  --max-steps 1 \
  --train-limit 24 \
  --validation-limit 8 \
  --test-limit 8
```

Example A100 run:

```bash
python lora-fine-tuning/methods/02_causal_lm_generation_parsing/qwen3_0.6b_generation_parsing_sweep.py run-sweep --resume
```

## 03 Causal LM: Next-Token Scoring

Trains the same causal-LM prompt/completion task, but evaluates by comparing the next-token logits for the single-token labels `ham` and `spam`. This avoids free-form generation parsing.

- Training notebook: `03_causal_lm_next_token/notebooks/qwen3_0.6b_casual_lm_next_token.ipynb`
- Sweep notebook: `03_causal_lm_next_token/notebooks/qwen3_0.6b_casual_lm_next_token_sweep.ipynb`
- Checkpoint eval notebook: `03_causal_lm_next_token/notebooks/qwen3_0.6b_checkpoint_200_test_eval.ipynb`
- Canonical sweep script remains at `../qwen3_0.6b_casual_lm_sweep.py`
- New method-local results can go under `03_causal_lm_next_token/results/`

The completed A100 sweep you analyzed remains at the repository-level `results/qwen3_clm_sweep_20260506_233813/`.

## 04 Causal LM: Structured JSON Generation

Changes the causal-LM training target from plain label text to structured JSON:

```json
{"label":"spam"}
```

This tests whether a generation-oriented training format reduces malformed outputs and parse failures while still using generation + parsing at inference time.

- Sweep script: `04_causal_lm_structured_generation/qwen3_0.6b_structured_generation_sweep.py`
- Sweep notebook: `04_causal_lm_structured_generation/notebooks/qwen3_0.6b_structured_generation_sweep.ipynb`
- New results: `04_causal_lm_structured_generation/results/`

Example A100 run:

```bash
python lora-fine-tuning/methods/04_causal_lm_structured_generation/qwen3_0.6b_structured_generation_sweep.py run-sweep --resume
```
