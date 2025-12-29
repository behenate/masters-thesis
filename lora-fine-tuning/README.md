# Qwen 1.5b fine-tuning

Initial attempt at fine-tuning Qwen 1.5b with quantization.

# Usage

Create and activate the python environment. Install dependencies.

```
# Create the environment
python3 -m venv .venv

# Activate the environment
source ./.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

In order for vscode to see the .venv for the jupyter notebook you may have to:

- Press `File` -> `Add folder to workspace` -> Select the lora fine tuning folder
- Select the jupyter environment `Cmd+Shift+P` -> Search for `Jupyter environment`
