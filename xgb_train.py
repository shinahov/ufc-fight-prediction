import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------- Load & Clean -----------------
df = pd.read_csv("snapshot.csv")
df["event_date"] = pd.to_datetime(df["event_date"])

max = df["event_date"].max()
delta = (max - df["event_date"]).dt.days

half_life = 365.0  # 1 Jahr
wights = np.power(2.0, -delta / half_life)

wights /= wights.mean()

df["weight"] = wights


# Nur Kämpfer mit Historie + bekanntem Sieger
df = df[(df["prior__fights"] > 0) & (df["winner"].notna())].copy()
grp = df.groupby("fight_order")["side"].nunique()
valid_ids = grp[grp == 2].index
df = df[df["fight_order"].isin(valid_ids)].copy()

# 3) Kontrolle
print(df["side"].value_counts())

print(df[["fighter_a", "fighter_b", "winner"]].head(40))


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

prior_cols = [c for c in df.columns if c.startswith("prior_")]

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
print((y == 0).sum() / (y == 1).sum())
groups = A["fight_order"].to_numpy()

# Klassenbalance anzeigen
vals, cnts = np.unique(y, return_counts=True)
print("Label-Verteilung (A gewinnt?):", dict(zip(vals, np.round(cnts / cnts.sum(), 3))))

# ----------------- Grouped Train/Val/Test Split -----------------
# Anzahl Gesamtzeilen
n = len(X)

# Prozentuale Aufteilung wie oben (Train/Val/Test)
train_size = 0.80
val_size   = 0.10
test_size  = 0.10

# Splitpunkte
tr_end = int(train_size * n)
va_end = int((train_size + val_size) * n)

# Datensätze in Reihenfolge (kein Shuffle)
X_tr, y_tr = X.iloc[:tr_end], y[:tr_end]
X_va, y_va = X.iloc[tr_end:va_end], y[tr_end:va_end]
X_te, y_te = X.iloc[va_end:], y[va_end:]

print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")


X_tr_f = X_tr.fillna(0.0).to_numpy(np.float32)
X_va_f = X_va.fillna(0.0).to_numpy(np.float32)
X_te_f = X_te.fillna(0.0).to_numpy(np.float32)
y_tr_f = y_tr.astype(np.float32)
y_va_f = y_va.astype(np.float32)
y_te_f = y_te.astype(np.float32)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr_f).astype(np.float32)
X_va_s = scaler.transform(X_va_f).astype(np.float32)
X_te_s = scaler.transform(X_te_f).astype(np.float32)

def train_eval_latent(k, X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te):
    tf.keras.backend.clear_session()
    tf.random.set_seed(42); np.random.seed(42)

    in_dim = X_tr_s.shape[1]
    inp = layers.Input((in_dim,), name=f"inp_{k}")
    h = layers.Dense(128, activation="relu", name=f"h_{k}")(inp)
    z  = layers.Dense(k, name=f"z_{k}")(h)

    d = layers.Dense(128, activation="relu", name=f"d_{k}")(z)
    x_hat = layers.Dense(in_dim, name=f"x_hat_{k}")(d)

    c = layers.Dense(64, activation="relu", name=f"c_{k}")(z)
    y_hat = layers.Dense(1, activation="sigmoid", name=f"y_hat_{k}")(c)

    model = keras.Model(inp, [x_hat, y_hat], name=f"ae_cls_{k}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"x_hat_"+str(k): "mse", "y_hat_"+str(k): "binary_crossentropy"},
        loss_weights={"x_hat_"+str(k): 1.0, "y_hat_"+str(k): 2.0},
        metrics={"y_hat_"+str(k): [keras.metrics.AUC(name="auc"), "accuracy"]},
    )
    cb = [keras.callbacks.EarlyStopping(monitor=f"val_y_hat_{k}_auc", mode="max",
                                        patience=15, restore_best_weights=True)]
    model.fit(
        X_tr_s, {"x_hat_"+str(k): X_tr_s, "y_hat_"+str(k): y_tr.astype(np.float32)},
        validation_data=(X_va_s, {"x_hat_"+str(k): X_va_s, "y_hat_"+str(k): y_va.astype(np.float32)}),
        epochs=300, batch_size=256, verbose=0, callbacks=cb
    )

    # Probas
    p_va = model.predict(X_va_s, verbose=0)[1][:,0]
    p_te = model.predict(X_te_s, verbose=0)[1][:,0]

    # Threshold aus VAL optimieren (Accuracy; alternativ F1/Youden)
    ts = np.linspace(0.3, 0.7, 81)
    accs = [((p_va>=t)==y_va).mean() for t in ts]
    t_best = ts[int(np.argmax(accs))]

    # Metriken
    out = {
        "k": k,
        "val_auc": float(roc_auc_score(y_va, p_va)),
        "val_acc@best": float(((p_va>=t_best)==y_va).mean()),
        "t_best": float(t_best),
        "test_auc": float(roc_auc_score(y_te, p_te)),
        "test_acc@best": float(((p_te>=t_best)==y_te).mean()),
    }
    return out, model

results = []
best = None
best_val = -1

for k in [4, 6, 8, 12, 16]:
    res, mdl = train_eval_latent(k, X_tr_s, y_tr_f, X_va_s, y_va_f, X_te_s, y_te_f)
    results.append(res)
    if res["val_auc"] > best_val:
        best_val = res["val_auc"]
        best = {"k": k, "res": res, "model": mdl}

print("All:", results)
print("Best (by VAL AUC):", best["res"])



print("Train/Val/Test sizes:", X_tr.shape, X_va.shape, X_te.shape)
print("XGb boost results:")

# ----------------- Model & Training -----------------
model = xgb.XGBClassifier(
    scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
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

n = len(X_tr)
w_tr = np.linspace(0.1, 5.0, n).astype(np.float32)



model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    eval_set=[(X_va, y_va)],
    verbose=False
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
top = sorted(imp_gain.items(), key=lambda x: x[1], reverse=True)#[:40]
print("Top-Features (gain):")
for k, v in top:
    print(k, round(v, 3))

model.save_model("xgb_fight_model.json")