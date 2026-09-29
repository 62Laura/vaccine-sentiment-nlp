
import os, time, random, copy
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# %% Load data (ONLY the three processed files, never re-split)
DATA = "data/processed"
train = pd.read_csv(f"{DATA}/train.csv")
val = pd.read_csv(f"{DATA}/val.csv")
test = pd.read_csv(f"{DATA}/test.csv")
for df in (train, val, test):
    df["text_clean"] = df["text_clean"].fillna("").astype(str)

# Sanity check against the handoff contract
assert len(train) == 7000 and len(val) == 1500 and len(test) == 1500, "wrong split sizes!"
print(train.label.value_counts(normalize=True).sort_index())

# %% Tokenize from text_clean; vocab built from TRAIN ONLY (no leakage)
MIN_FREQ = 2          # words seen once in train -> <unk>. Handles the 65% singleton tail.
PAD, UNK = 0, 1

counts = Counter(tok for t in train.text_clean for tok in t.split())
itos = ["<pad>", "<unk>"] + [w for w, c in counts.items() if c >= MIN_FREQ]
stoi = {w: i for i, w in enumerate(itos)}
print(f"vocab size: {len(itos)} (from {len(counts)} unique train tokens)")

lengths = train.text_clean.str.split().str.len()
MAX_LEN = int(np.percentile(lengths, 95))
print("MAX_LEN (95th pct):", MAX_LEN)

def encode(texts):
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, UNK) for w in t.split()][:MAX_LEN]
        out[i, :len(ids)] = ids
    return torch.tensor(out)

Xtr, Xva, Xte = encode(train.text_clean), encode(val.text_clean), encode(test.text_clean)
ytr, yva, yte = (torch.tensor(d.label.values) for d in (train, val, test))

BATCH = 64
train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(TensorDataset(Xva, yva), batch_size=256)
test_dl = DataLoader(TensorDataset(Xte, yte), batch_size=256)

# %% Class weights (imbalance: negative is ~10%). Report this in the notes column.
cnt = np.bincount(train.label.values, minlength=3)
class_w = torch.tensor(len(train) / (3 * cnt), dtype=torch.float).to(DEVICE)
print("class weights [neg, neu, pos]:", class_w.cpu().numpy().round(2))

# %% Model definitions
NUM_CLASSES, EMB_DIM = 3, 100

class BiLSTM(nn.Module):
    def __init__(self, vocab, emb=EMB_DIM, hidden=128, drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden * 2, NUM_CLASSES)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)
        h, _ = self.lstm(self.drop(self.emb(x)))
        h = h.masked_fill(~mask, -1e9).max(dim=1).values   # max-pool over real tokens only
        return self.fc(self.drop(h))

class TextCNN(nn.Module):
    def __init__(self, vocab, emb=EMB_DIM, n_filters=100, ks=(2, 3, 4), drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb, n_filters, k) for k in ks)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(n_filters * len(ks), NUM_CLASSES)

    def forward(self, x):
        e = self.drop(self.emb(x)).transpose(1, 2)          # (B, emb, L)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))

# %% Train / eval helpers
def predict(model, dl):
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
    return torch.cat(preds).numpy()

def train_model(model, epochs=20, lr=1e-3, patience=4):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    best_f1, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item() * len(xb)
        val_f1 = f1_score(yva.numpy(), predict(model, val_dl), average="macro")
        print(f"epoch {ep:2d} | loss {total/len(train):.4f} | val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("early stop"); break
    train_time = time.time() - t0
    model.load_state_dict(best_state)      # restore best epoch (chosen on VAL, not test)
    return train_time

def evaluate_and_log(model, name, train_time, notes):
    y_true, y_pred = yte.numpy(), predict(model, test_dl)
    row = {
        "model_name": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro"), 3),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 3),
        "train_time_s": round(train_time, 2),
        "notes": notes,
    }
    print(row)

    os.makedirs("results/confusion_matrices", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"results/confusion_matrices/{name}.png", dpi=150)
    plt.show()

    path = "results/metrics_summary.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return row

# %% Model 3: BiLSTM
NOTES = f"emb={EMB_DIM} random init, min_freq={MIN_FREQ}, max_len={MAX_LEN}, class-weighted loss, early stop on val macro-F1"
m3 = BiLSTM(len(itos))
t3 = train_model(m3)
evaluate_and_log(m3, "model3_bilstm", t3, "BiLSTM(128) max-pool; " + NOTES)

# %% Model 4: Text CNN
m4 = TextCNN(len(itos))
t4 = train_model(m4)
evaluate_and_log(m4, "model4_textcnn", t4, "CNN k=2,3,4 x100 filters; " + NOTES)
