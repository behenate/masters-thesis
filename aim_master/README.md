# Aim Master Tools

This directory contains a small local toolkit for working with downloaded Aim repositories.

It gives you two safe workflows:

- merge downloaded `.aim` repositories into one master repo
- pick a run from the master repo and export its metrics to CSV

The master repo lives at:

`aim_master/master_repo`

Exported CSV files are written to:

`aim_master/exports`

## Install

If you already use the project `.venv`, reinstall the package with the Aim master extra:

```bash
./.venv/bin/pip install -e '.[aim-master]' --no-build-isolation
```

## Open master repo dashboard

```bash
# Run assuming the current pwd is the root of the project
aim up --repo "./aim_master/master_repo"
```

## Merge Repositories

Run:

```bash
aim-master-merge
```

What it does:

- initializes `aim_master/master_repo` if it does not exist yet
- opens a directory picker so you can choose downloaded Aim repos one by one
- safely merges them with Aim's own CLI, copying runs one by one into the master repo

You can also pass sources directly:

```bash
aim-master-merge --source /path/to/repo1 --source /path/to/repo2
```

If you want to move runs instead of copying them:

```bash
aim-master-merge --move
```

## Export Metrics To CSV

Run:

```bash
aim-master-export
```

What it does:

- reads runs from `aim_master/master_repo`
- shows a keyboard-driven multi-run picker
- always includes a default metric set when available
- then shows the remaining metrics so you can select extras with the keyboard
- exports the result to a CSV in `aim_master/exports`

By default, the CSV stays lean and metric-focused. Each row contains only:

- `run_hash`
- `metric_name`
- `context_label`
- `step`
- `epoch`
- `time`
- `value`

This makes the output much easier to analyze in a notebook, spreadsheet, or plotting tool.

Default metrics include common training, evaluation, and system signals such as:

- `loss`
- `learning_rate`
- `grad_norm`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `balanced_accuracy`
- `roc_auc`
- `pr_auc`
- `cross_entropy`
- `brier_score`
- `matthews_corrcoef`
- `__system__cpu`
- `__system__memory_percent`

You can export every metric without the extra selection step:

```bash
./.venv/bin/aim-master-export --all-metrics
```

You can also target one run directly:

```bash
./.venv/bin/aim-master-export --run-hash YOUR_RUN_HASH
```

Or export several runs into one CSV:

```bash
./.venv/bin/aim-master-export --run-hash RUN_HASH_1 --run-hash RUN_HASH_2
```

Comma-separated hashes are accepted as well:

```bash
./.venv/bin/aim-master-export --run-hash RUN_HASH_1,RUN_HASH_2
```

And you can override the output path:

```bash
./.venv/bin/aim-master-export --output /tmp/my_run_metrics.csv
```

If you want the expanded run metadata columns as well, opt in explicitly:

```bash
./.venv/bin/aim-master-export --include-run-metadata
```

## Notes

- Do not manually copy files into `.aim` directories.
- The merge tool uses Aim’s own CLI for safe copying.
- If the directory picker is unavailable, the merge tool falls back to manual path entry in the terminal.
- The exporter reads metric data from the original downloaded source repo recorded in the manifest, while the master repo is used for run selection and organization.
