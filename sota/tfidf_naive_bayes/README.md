# TF-IDF + Multinomial Naive Bayes

Baseline based on TF-IDF text representation and a Multinomial Naive Bayes
classifier.

Run from the repository root:

```bash
./.venv/bin/python sota/tfidf_naive_bayes/run_tfidf_naive_bayes.py
```

The script trains on the `dataset.combine` `training_all` 92% train split
using seed 67, then evaluates `train_subset`, `enron`,
`fraudulent_email_corpus`, and `spam_ham` with a 2000-row sample limit.
It writes `sota/tfidf_naive_bayes/summary.csv`.
