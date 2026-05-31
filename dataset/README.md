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

`training_all` expands to all registered datasets except `enron`, `fraudulent_email_corpus`, and `spam_ham`.

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
