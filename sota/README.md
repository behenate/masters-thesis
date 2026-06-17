# State-of-the-art baselines

This directory contains independent baseline runs used to compare the thesis
model with established text-classification methods.

Each method should evaluate the same datasets used in the thesis experiments:

- `train_subset`
- `enron`
- `fraudulent_email_corpus`
- `spam_ham`

Default evaluation settings:

- sample limit: `2000`
- seed: `67`
- output file: `summary.csv`

Expected metrics should match the thesis evaluation tables whenever possible:
`accuracy`, `precision`, `recall`, `f1`, `specificity`,
`balanced_accuracy`, false-positive and false-negative counts/rates, and
runtime.

Tune the classical and frozen-embedding methods on the 2% validation split:

```bash
./.venv/bin/python sota/tune_baselines.py
```

The tuning script selects one configuration per method without using the three
external evaluation datasets. It writes `tuning_results.csv`,
`tuned_config.json`, the persisted model artifact, and the final `summary.csv`
inside each method directory.

Re-evaluate the persisted selected models without repeating the search:

```bash
./.venv/bin/python sota/tune_baselines.py --evaluate-only
```
