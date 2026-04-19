# Dataset Tools

Use [combine.py](/Users/wojciechdrozdz/uni/Magisterka/Semestr%20III/Praca%20Magisterska/dataset/combine.py) to download, normalize, deduplicate, balance, and save merged datasets.

## Combining datasets

```python
from dataset.combine import combine_datasets

dataset_path = combine_datasets(
    ["trec_2007", "ceas_2008"],
    spam_ham_ratio=0.5,
    duplicate_detection="high",
    generate_duplicate_report=False,
)
```

The output path is deterministic for the same:

- dataset set
- `spam_ham_ratio`
- `duplicate_detection`

So repeated calls reuse the same cached parquet file.

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
