# Vaccine Sentiment Classification — Sequential Models (Formative 2)

Group project for *Research-Informed Sequential Models for NLP and Language Technologies*.
Dataset: Zindi — [To Vaccinate or Not to Vaccinate?](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate) (tweet sentiment: negative / neutral / positive).

**Read `HANDOFF.md` before writing any model code** — it has the locked data split, schema, real EDA numbers, and the exact metrics format every model must report.

## Repo structure

```
vaccine-sentiment-nlp/
├── README.md
├── HANDOFF.md                         ← the contract — read this first
├── requirements.txt
├── data/
│   ├── raw/                           ← original Zindi CSV goes here (not committed if large)
│   └── processed/
│       ├── train.csv                  ← add this (see HANDOFF.md)
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← Person 1 (done)
│   ├── 02_related_work.md             ← Person 2
│   ├── 03b_model2_baseline.ipynb      ← Person 2
│   ├── 03c_model3_bilstm.ipynb        ← Person 3
│   ├── 03d_model4_textcnn.ipynb       ← Person 3
│   └── 03e_model5_distilbert.ipynb    ← Person 4
├── results/
│   ├── metrics_summary.csv            ← every person appends one row, same columns
│   └── confusion_matrices/
├── figures/
│   └── eda/
└── contribution_tracker.md
```

## Task split

| Person | Role | Deliverable |
|---|---|---|
| 1 | Data Explorer | EDA + Model 1 (TF-IDF + Logistic Regression) — **done, this notebook** |
| 2 | Researcher | Literature review + Model 2 (TF-IDF + Naive Bayes or SVM) |
| 3 | Neural Model Builder A | Model 3 (BiLSTM) + Model 4 (Text CNN) |
| 4 | Neural Model Builder B + Results Manager | Model 5 (fine-tuned DistilBERT) + final comparison tables/confusion matrices + repo cleanup |

## Status

- [x] Data downloaded and explored (`notebooks/01_data_exploration.ipynb`)
- [x] Cleaned data split into train/val/test (`data/processed/`) — **train.csv still needs to be added**
- [x] Model 1 baseline trained and logged
- [ ] Related work review
- [ ] Models 2–5
- [ ] Error analysis
- [ ] Final report + demo video

## Setup

```bash
pip install -r requirements.txt
```
