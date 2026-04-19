# Detecting Spam Mail and Phishing using Large Language Models

This repository contains all files used to develop e-mail spam detection using fine-tuning of small LLMs. This `README` will be updated in sync with the progress of the project.

## Setup

### Local deployment

Use the repository setup script to create a virtual environment and install the dataset utilities plus the LoRA fine-tuning stack:

```bash
./setup.sh
source .venv/bin/activate
```

This installs the project in editable mode together with the dependencies needed by the fine-tuning notebooks, including:

- `torch`
- `transformers`
- `peft`
- `datasets`
- `accelerate`
- `jupyterlab`

### Cloud GPU deployment

On a CUDA machine, install the NVIDIA CUDA toolkit first. The helper script below checks for `nvcc`, adds CUDA paths to `~/.bashrc`, and then runs the shared Python setup with GPU-specific extras such as `flash-attn`.
Script is compatible with Ubuntu 22.04

```bash
./setup_thunder_compute.sh
source .venv/bin/activate
```

If CUDA is not installed yet, the script will stop and print the NVIDIA download page you should use first.

### Manual install command

If you already have an activated virtual environment and want the package install directly:

```bash
pip install -e ".[lora-fine-tuning]" --no-build-isolation
```

For GPU environments that should also install `flash-attn`, run:

```bash
INSTALL_FLASH_ATTN=1 ./setup.sh
```

### Aim

The repository uses Aim to track the training parameters. In order to start a web interface run `./.venv/bin/aim up`

To expose the Aim dashboard through ngrok and print a shortened public link, run:

```bash
./share_aim_dashboard.sh
```

The script uses `lnk.ua` by default for shortening.
If you want the script to configure ngrok automatically, create a root `.env` file from `.env.template` and set `NGROK_AUTHTOKEN`.

If you want to keep the raw ngrok URL instead, run:

```bash
SHORTEN_PUBLIC_URL=0 ./share_aim_dashboard.sh
```

## Dataset combining

The dataset utilities expose `combine_datasets(...)` from [dataset/combine.py](/Users/wojciechdrozdz/uni/Magisterka/Semestr%20III/Praca%20Magisterska/dataset/combine.py).

Example:

```python
from dataset.combine import combine_datasets

dataset_path = combine_datasets(
    ["trec_2007", "ceas_2008"],
    spam_ham_ratio=0.5,
    duplicate_detection="high",
    generate_duplicate_report=False,
)
```

Duplicate detection modes:

- `basic`: the original exact duplicate rule using `subject + body + label`
- `medium`: deduplicate by `body` only, with whitespace removed
- `high`: the strongest mode, deduplicating by aggressively normalized `body`

`high` is the default.

If you want a CSV describing the duplicate groups that were removed, enable:

```python
combine_datasets(
    ["spam_assassin", "spam_ham"],
    duplicate_detection="high",
    generate_duplicate_report=True,
)
```

The duplicate report stores every sample in each duplicate group in wide columns such as:

- `sample_0_subject`
- `sample_0_body`
- `sample_0_label`
- `sample_0_source`
- `sample_1_subject`
- `sample_1_body`

and so on for every sample in the group.
