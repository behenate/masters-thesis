# TF-IDF + Logistic Regression

Baseline based on TF-IDF text representation and logistic regression.

## Parameter selection

`sota/tune_baselines.py` compared `C` values `0.01`, `0.1`, `1`, and `10` and
selected the classification threshold on the 2% validation split. The selected
configuration uses `C=10` and threshold `0.3579048`. The external datasets were
used only for the final evaluation and did not affect parameter selection.
