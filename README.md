# ml-03-classification-projects

My solutions to the Chapter 3 (Classification) exercises of *Hands-On Machine Learning with Scikit-Learn and PyTorch*.
The chapter's own project (an MNIST classifier with a full evaluation report) lives in the
**`hands-on-ml`** repository under `part1/ch03-classification/`.

| Exercise | What it does | Result | File |
|---|---|---|---|
| 1–2 | Tune KNN on MNIST to beat 97%, then add 4 one-pixel-shifted copies of every training image | 97.14% test accuracy (distance-weighted, k=4) → 97.63% with augmentation | [`ex01-02-mnist-97-and-augmentation.ipynb`](ex01-02-mnist-97-and-augmentation.ipynb) |
| 3 ★ | Titanic survival: feature engineering (title, family size), 4 model families, tuning | 83.7% 10-fold CV accuracy (tuned SVC) | [`ex03-titanic/`](ex03-titanic/) |
| 4 ★ | Spam filter on the SpamAssassin corpus: email parsing, configurable word-count transformer, sparse vectors | test precision 99.0%, recall 95.0% | [`ex04-spam-filter/`](ex04-spam-filter/) |

★ = project-sized exercise with its own README.

This chapter has no theory questions.

## Run
```bash
uv sync                # creates .venv with everything (Python 3.12 recommended)
uv run jupyter lab
```
Datasets are downloaded on first use into `datasets/` (git-ignored): MNIST from OpenML, Titanic from the
book's data repository, the spam corpus from spamassassin.apache.org. Set `HOML_DATA_DIR` to use another folder.

Run times on a 2-CPU machine: Exercises 1–2 take the longest (the augmented KNN predicts against 300,000 stored images).

---
Based on the exercises in *Hands-On Machine Learning with Scikit-Learn and PyTorch* by Aurélien Géron (O'Reilly, 2025). All code written from scratch.
