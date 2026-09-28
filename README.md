# Vaccine Sentiment Classification — Sequential Models (Formative 2)

Group project for *Research-Informed Sequential Models for NLP and Language Technologies*.
Dataset: Zindi — [To Vaccinate or Not to Vaccinate?](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate) (tweet sentiment: negative / neutral / positive).


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

## Setup

```bash
pip install -r requirements.txt
```
