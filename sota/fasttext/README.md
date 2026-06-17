# fastText supervised classifier

Baseline using fastText supervised text classification.

## Parameter selection

`sota/tune_baselines.py` used a two-stage search. The first stage compared
learning rates `0.05`, `0.1`, and `0.25` and epoch counts `5`, `10`, and `20`.
The second stage used the best optimization settings to compare `wordNgrams`
values `1`, `2`, and `3`, dimensions `50`, `100`, and `200`, and character
n-grams disabled or set to `minn=3`, `maxn=6`. Every run used seed `67` and
eight threads.

The selected model uses `lr=0.25`, `epoch=20`, `wordNgrams=1`, `dim=100`,
character n-grams `(3, 6)`, and spam threshold `0.2517502`. Parameter selection
used only the validation split; the three external datasets remained final
test sets.
