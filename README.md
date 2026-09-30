# Vaccine Sentiment Classification — Sequential Models (Formative 2)

Group project for *Research-Informed Sequential Models for NLP and Language Technologies*.

We used the Zindi dataset [To Vaccinate or Not to Vaccinate?](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate), which contains tweets labeled by sentiment toward vaccination: negative, neutral, or positive. Our goal was to compare five different modelling approaches on the same data and same train/val/test split, and evaluate which ones handle this kind of short, noisy social media text best.

## Dataset

- Source: Zindi "To Vaccinate or Not to Vaccinate?" challenge
- Task: 3-class sentiment classification (negative / neutral / positive)
- Raw data: 10,000 labeled tweets (after removing rows with missing labels)
- Split: 7,000 train / 1,500 validation / 1,500 test, kept identical across all five models so results are directly comparable
- Labels were originally -1/0/1 in the raw file and remapped to 0/1/2 (negative/neutral/positive) for consistency

## Models Compared

| Model | Type | Description |
|---|---|---|
| TF-IDF + Logistic Regression | Baseline | TF-IDF unigrams and bigrams, 10k features, class-weighted |
| TF-IDF + Linear SVM | Baseline | TF-IDF features, hyperparameters tuned with grid search |
| BiLSTM | Neural | Bidirectional LSTM with max-pooling, randomly initialized embeddings |
| Text CNN | Neural | Convolutional filters over word embeddings (kernel sizes 2, 3, 4) |
| DistilBERT | Neural | Fine-tuned pretrained transformer (`distilbert-base-uncased`) |

## Results

| Model | Accuracy | Macro F1 | Precision (macro) | Recall (macro) |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.727 | 0.664 | 0.654 | 0.692 |
| TF-IDF + Linear SVM | 0.717 | 0.634 | 0.633 | 0.635 |
| BiLSTM | 0.712 | 0.638 | 0.632 | 0.647 |
| Text CNN | 0.671 | 0.603 | 0.596 | 0.625 |
| DistilBERT | 0.755 | 0.669 | 0.679 | 0.661 |

DistilBERT achieved the best accuracy and macro F1 overall, though only slightly ahead of the simple Logistic Regression baseline. Full comparison chart: `results/model_comparison_macro_f1.png`. Full metrics table: `results/metrics_summary.csv`.

## Repo Structure

vaccine-sentiment-nlp/
├── README.md
├── HANDOFF.md contract notes for the group
├── requirements.txt
├── data/
│ ├── raw/
│ │ └── Train.csv original Zindi data (10,001 rows, includes label + agreement)
│ └── processed/
│ ├── val.csv cleaned validation set with labels
│ └── test.csv cleaned test set with labels
├── notebooks/
│ ├── 01_data_exploration.ipynb EDA and data cleaning
│ ├── 02_related_work.md literature review
│ ├── 03b_model2_baseline.ipynb TF-IDF + SVM
│ ├── models_3_4_bilstm_cnn.py BiLSTM and Text CNN
│ └── 03e_model5_distilbert.ipynb DistilBERT fine-tuning
├── results/
│ ├── metrics_summary.csv one row per model, same columns
│ ├── model_comparison_macro_f1.png bar chart comparing all 5 models
│ ├── svm_errors.csv misclassified examples from SVM
│ └── confusion_matrices/
│ ├── cm_model1.png
│ ├── tfidf_linear_svm.png
│ ├── model3_bilstm.png
│ ├── model4_textcnn.png
│ └── model5_distilbert.png
├── figures/
│ └── eda/ class distribution, length distribution, etc.
└── contribution_tracker.md


## Setup

```bash
pip install -r requirements.txt
```

Notebooks for the neural models (BiLSTM, Text CNN, DistilBERT) were run on Google Colab with a T4 GPU. The DistilBERT notebook in particular needs a GPU runtime to train in a reasonable time.

## Team Contributions

See `contribution_tracker.md` for a full breakdown. In short:

- **Laura Celine** — Data cleaning, EDA, train/val/test split, TF-IDF + Logistic Regression baseline
- **Sonia Uwase** — Literature review, TF-IDF + Linear SVM baseline
- **Heroine Mutumwinka** — BiLSTM and Text CNN models
- **Hasbiyallah Umutoniwabo** — DistilBERT fine-tuning, model comparison, results collection

## Notes on Academic Integrity

AI tools were used to assist with debugging, and drafting parts of this README, as disclosed in the final report. All modelling decisions, experiments, and analysis were reviewed and understood by the team.
