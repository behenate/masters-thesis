# Dataset Tools

Use [combine.py](/Users/wojciechdrozdz/uni/Magisterka/Semestr%20III/Praca%20Magisterska/dataset/combine.py) to download, normalize, deduplicate, balance, and save merged datasets.

## Combining datasets

```python
from dataset.combine import combine_datasets

dataset_path = combine_datasets(
    "training_all",
    combination_mode="mixed_50_50",
    duplicate_detection="high",
    generate_duplicate_report=False,
)
```

The output path is deterministic for the same:

- dataset set
- `spam_ham_ratio`
- `duplicate_detection`
- `combination_mode`
- `source_aware_max_multiplier` for `source_aware_50_50`

So repeated calls reuse the same cached parquet file.

`training_all` expands to all registered datasets except `enron`, `fraudulent_email_corpus`, `spam_ham`, and `trec_2006`.
The `trec_2006` corpus is reserved for final external evaluation and must not be used for training or model selection.

## TREC 2006 overlap analysis

After downloading the datasets, compare TREC 2006 with every previously registered source:

```bash
python dataset/analyze_trec_2006_duplicates.py
```

The script writes `summary.csv`, `matches.csv`, `metrics.json`, and the deduplicated
`eligible_pool.parquet` under `dataset/reports/trec_2006_overlap`. Labels are not
included in duplicate fingerprints. Exact matching uses normalized whitespace over
the subject and body. Longer messages are additionally compared with the existing
`high` body normalization; short messages require an exact subject-and-body match to
avoid accidental matches such as `test` or `help`.

## Frozen evaluation splits

Prepare the historical external-validation samples and disjoint final-test samples
directly from the public source datasets:

```bash
python dataset/prepare_evaluation_splits.py
```

The default output is `dataset/evaluation_splits/seed_67`. Its `manifest.json`
contains row counts, class counts, source hashes, Parquet hashes, content hashes,
and leakage checks. The final tests contain another 1000 records from `enron`,
`fraudulent_email_corpus`, and `spam_ham`, plus a balanced 1000-record TREC 2006
test. Final-test fingerprints are checked against the training dataset, external
validation samples, and the other final-test sets.

Evaluate the frozen final-test splits without resampling them:

```bash
python lora-fine-tuning/methods/03_causal_lm_next_token/notebooks/evaluate_checkpoints.py \
  --method 03 \
  --dataset-manifest dataset/evaluation_splits/seed_67/manifest.json \
  --split-roles final_test \
  --batch-size 16
```

Use `--split-roles external_validation final_test` to evaluate both roles in one
run. The same evaluator supports methods `01`, `02`, `03`, and `04` through
`--method`.

## Combination modes

- `mixed`
  Default backwards-compatible behavior. Mixes all requested datasets and applies `spam_ham_ratio`; pass `0.5` for a global 50/50 spam/ham split.

- `mixed_50_50`
  Mixes all requested datasets, then trims only by class to produce the largest possible 50/50 spam/ham split.

- `source_aware_50_50`
  Produces a 50/50 spam/ham split with capped square-root source allocation. Each source/class bucket is capped at `source_aware_max_multiplier` times the smallest bucket for that class, defaulting to `8.0`, so smaller datasets keep more relative weight and very large datasets are trimmed harder.

- `balanced_source_50_50`
  Takes the same number of ham and spam samples from every source. The per-source, per-class count is limited by the smallest source/class bucket.

## Duplicate detection modes

- `basic`
  Exact duplicate removal using `subject + body + label`.

- `medium`
  Deduplicate by `body` only after removing whitespace.

- `high`
  Deduplicate by aggressively normalized `body`. This mode normalizes Unicode, decodes HTML entities, removes HTML tags, strips URLs and email addresses, lowercases text, and removes non-alphanumeric characters.

The saved parquet keeps one original row unchanged. Only duplicate detection uses the normalized fingerprint.

## Duplicate reports

If you pass `generate_duplicate_report=True`, the combiner also writes a CSV report under [dataset/reports](/Users/wojciechdrozdz/uni/Magisterka/Semestr%20III/Praca%20Magisterska/dataset/reports).

Each report row describes one duplicate group and includes:

- `duplicate_count`
- `labels`
- `sources`
- the kept row metadata
- `all_original_row_indexes`
- `sample_0_original_row_index`
- `sample_0_subject`
- `sample_0_body`
- `sample_0_label`
- `sample_0_source`

The report continues with `sample_1_*`, `sample_2_*`, and so on for every sample in the duplicate group, so each original duplicate row is available directly as CSV columns.
