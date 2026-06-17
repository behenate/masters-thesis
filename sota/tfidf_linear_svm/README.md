# TF-IDF + Linear SVM

Baseline based on TF-IDF text representation and a linear support-vector
classifier.

## Parameter selection

`sota/tune_baselines.py` compared word n-grams `(1, 2)` with character n-grams
`(3, 5)`, used `max_df=0.95`, and tested `C` values `0.01`, `0.1`, `1`, and
`10`. The selected configuration uses character n-grams, `C=10`, and decision
threshold `0.1031114`. The external datasets were used only after selecting the
configuration on the validation split.
