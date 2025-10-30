import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from xgboost.callback import EarlyStopping

# ----------------- Load & Clean -----------------
df = pd.read_csv("snapshot.csv")

# Nur Kämpfer mit Historie + bekanntem Sieger
df = df[(df["prior__fights"] > 0) & (df["winner"].notna())].copy()

# Sicherstellen: pro fight_order genau 2 Zeilen (A & B)
cnt = df.groupby("fight_order").size()
valid_fights = cnt[cnt == 2].index
df = df[df["fight_order"].isin(valid_fights)].copy()

# Label pro FIGHT aus Sicht A (1 = A gewinnt)
df["label"] = (df["winner"] == df["fighter"]).astype(int)

# ----------------- Build Pairwise Features -----------------
# Split nach Seite und nach fight_order sortieren → 1:1 Alignment
A = df[df["side"] == "A"].sort_values("fight_order").reset_index(drop=True)
B = df[df["side"] == "B"].sort_values("fight_order").reset_index(drop=True)

# Sanity: gleiche Fight-IDs?
assert np.array_equal(A["fight_order"].values, B["fight_order"].values), "A/B nicht ausrichtbar."

prior_cols = [c for c in df.columns if c.startswith("prior__")]

A_vals = A[prior_cols].to_numpy(dtype=np.float32)
B_vals = B[prior_cols].to_numpy(dtype=np.float32)

# Differenz & Ratio (robust, vermeidet div/0 und inf)
X_diff  = A_vals - B_vals
X_ratio = (A_vals + 1.0) / (B_vals + 1.0)
X_ratio = np.clip(X_ratio, 1e-6, 1e6)

X = pd.DataFrame(
    np.hstack([X_diff, X_ratio]),
    columns=[f"diff::{c}" for c in prior_cols] + [f"ratio::{c}" for c in prior_cols]
).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

y = (A["winner"] == A["fighter"]).astype(int).to_numpy()
groups = A["fight_order"].to_numpy()

# Klassenbalance anzeigen
vals, cnts = np.unique(y, return_counts=True)
print("Label-Verteilung (A gewinnt?):", dict(zip(vals, np.round(cnts / cnts.sum(), 3))))

# ----------------- Grouped Train/Val/Test Split -----------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trval_idx, te_idx = next(gss.split(X, y, groups))
X_trval, X_te = X.iloc[trval_idx], X.iloc[te_idx]
y_trval, y_te = y[trval_idx], y[te_idx]
grp_trval = groups[trval_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
tr_idx, va_idx = next(gss2.split(X_trval, y_trval, grp_trval))
X_tr, X_va = X_trval.iloc[tr_idx], X_trval.iloc[va_idx]
y_tr, y_va = y_trval[tr_idx], y_trval[va_idx]

print("Train/Val/Test sizes:", X_tr.shape, X_va.shape, X_te.shape)

# ----------------- Model & Training -----------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=2.0,
    reg_alpha=0.5,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    early_stopping_rounds=100  # funktioniert bei 2.1.3
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    verbose=False,
)

# ----------------- Evaluation -----------------
def eval_set(Xs, ys, name):
    proba = model.predict_proba(Xs)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    print(f"{name}  Acc={accuracy_score(ys, pred):.3f}  AUC={roc_auc_score(ys, proba):.3f}")
    print(f"{name}  Confusion:\n{confusion_matrix(ys, pred)}")

eval_set(X_va, y_va, "VAL")
eval_set(X_te, y_te, "TEST")

# ----------------- Feature Importance (gain) -----------------
imp_gain = model.get_booster().get_score(importance_type="gain")
top = sorted(imp_gain.items(), key=lambda x: x[1], reverse=True)[:20]
print("Top-Features (gain):")
for k, v in top:
    print(k, round(v, 3))
