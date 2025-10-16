# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, optimizers, losses

df = pd.read_csv("final_df.csv").drop(columns=["A_Name", "B_Name"])

diff_df = {}
for col in df.columns:
    if col.startswith("A_"):
        suff = col[2:]
        b = "B_" + suff
        if b in df.columns:
            diff_df["diff_" + suff] = df[col] - df[b]
diff_df = pd.DataFrame(diff_df, index=df.index)

X_diff = diff_df.copy()
X_diff["time_format"] = df["time_format"].astype(str).str.contains("5").astype(int)
y_win = df["Winner_bin"].astype(int)

forbidden = {"result_score", "end_round", "end_time", "Winner_bin", "date"}
round_prefixes = tuple(f"r{i}_" for i in range(1, 6))
fight_stat_cols = [
    c for c in df.columns
    if ((c.startswith("a_") or c.startswith("b_"))
        and (not c.startswith(round_prefixes))
        and (c not in forbidden))
]
X_fs = df[fight_stat_cols].apply(pd.to_numeric, errors="raise")

idx_all = df.index
train_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=42, stratify=y_win)

X_fs_train = X_fs.loc[train_idx]
X_fs_test  = X_fs.loc[test_idx]
X_diff_train = X_diff.loc[train_idx]
X_diff_test  = X_diff.loc[test_idx]
y_train_win = y_win.loc[train_idx]
y_test_win  = y_win.loc[test_idx]

sc_fs = StandardScaler().fit(X_fs_train)
X_fs_train_s = sc_fs.transform(X_fs_train)
X_fs_test_s  = sc_fs.transform(X_fs_test)

sc_diff = StandardScaler().fit(X_diff_train)
X_diff_train_s = sc_diff.transform(X_diff_train)
X_diff_test_s  = sc_diff.transform(X_diff_test)

def train_autoencoder_with_winhead(
    X_train, y_train, X_val=None, y_val=None,
    *, latent_dim=5, noise_std=0.05,
    enc_sizes=(64, 32), dec_sizes=(32, 64),
    weight_decay=1e-4, lambda_win=0.25,
    lr=1e-3, batch_size=128, epochs=200, patience=12,
    seed=42, class_balance=True
):
    np.random.seed(seed); tf.random.set_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train).astype(int).ravel()
    if X_val is not None: X_val = np.asarray(X_val, dtype=np.float32)
    if y_val is not None: y_val = np.asarray(y_val).astype(int).ravel()

    input_dim = X_train.shape[1]
    reg = regularizers.l2(weight_decay)

    inp = layers.Input(shape=(input_dim,), name="input")
    x = layers.GaussianNoise(noise_std, name="noise")(inp)
    for i, hs in enumerate(enc_sizes):
        x = layers.Dense(hs, kernel_regularizer=reg, name=f"enc_dense_{i}")(x)
        x = layers.ReLU()(x)
    z = layers.Dense(latent_dim, name="z")(x)

    d = z
    for i, hs in enumerate(dec_sizes):
        d = layers.Dense(hs, kernel_regularizer=reg, name=f"dec_dense_{i}")(d)
        d = layers.ReLU()(d)
    recon = layers.Dense(input_dim, name="recon")(d)

    win_logits = layers.Dense(1, name="win")(z)

    model = models.Model(inp, outputs={"recon": recon, "win": win_logits})
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss={"recon": losses.MeanSquaredError(),
              "win":   losses.BinaryCrossentropy(from_logits=True)},
        loss_weights={"recon": 1.0, "win": float(lambda_win)},
    )

    es = callbacks.EarlyStopping(monitor="val_recon_loss", mode="min", patience=patience, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_recon_loss", mode="min", factor=0.5, patience=5, min_lr=1e-5)

    fit_kwargs = {}
    if class_balance and (len(np.unique(y_train)) == 2):
        classes = np.array([0, 1])
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        w0, w1 = float(cw[0]), float(cw[1])
        sw_win = np.where(y_train == 1, w1, w0).astype("float32")
        sw_recon = np.ones_like(sw_win, dtype="float32")
        fit_kwargs["sample_weight"] = {"recon": sw_recon, "win": sw_win}
    else:
        ones = np.ones(len(y_train), dtype="float32")
        fit_kwargs["sample_weight"] = {"recon": ones, "win": ones}

    if X_val is not None and y_val is not None:
        val_data = (X_val, {"recon": X_val, "win": y_val})
        history = model.fit(
            X_train, {"recon": X_train, "win": y_train},
            validation_data=val_data,
            epochs=epochs, batch_size=batch_size,
            callbacks=[es, rlrop], verbose=1, **fit_kwargs
        )
    else:
        history = model.fit(
            X_train, {"recon": X_train, "win": y_train},
            validation_split=0.2,
            epochs=epochs, batch_size=batch_size,
            callbacks=[es, rlrop], verbose=1, **fit_kwargs
        )

    encoder = models.Model(inp, z, name="encoder")
    z_in = layers.Input(shape=(latent_dim,), name="z_in")
    d = z_in
    for i, _ in enumerate(dec_sizes):
        d = model.get_layer(f"dec_dense_{i}")(d)
        d = layers.ReLU()(d)
    out_dec = model.get_layer("recon")(d)
    decoder = models.Model(z_in, out_dec, name="decoder")

    Z_train = encoder.predict(X_train, verbose=0)
    logits_train = model.predict(X_train, verbose=0)["win"].ravel()
    p_train = tf.math.sigmoid(logits_train).numpy()

    result = {"model": model, "encoder": encoder, "decoder": decoder, "history": history, "Z_train": Z_train, "p_train": p_train}

    if X_val is not None:
        Z_val = encoder.predict(X_val, verbose=0)
        logits_val = model.predict(X_val, verbose=0)["win"].ravel()
        p_val = tf.math.sigmoid(logits_val).numpy()
        result.update({"Z_val": Z_val, "p_val": p_val})

    return result

ae_res = train_autoencoder_with_winhead(
    X_fs_train_s, y_train_win.values,
    X_val=X_fs_test_s, y_val=y_test_win.values,
    latent_dim=5, noise_std=0.05,
    enc_sizes=(64, 32), dec_sizes=(32, 64),
    lambda_win=0.25, lr=1e-3, batch_size=128,
    epochs=200, patience=12, class_balance=True
)

Z_train = ae_res["Z_train"]
Z_test  = ae_res["Z_val"]

def train_z_regressor(X_train, Z_train):
    reg = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=400,
            max_depth=40,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    )
    reg.fit(X_train, Z_train)
    return reg

def evaluate_z_regressor(reg, X_test, Z_test):
    Z_pred = reg.predict(X_test)
    r2 = [r2_score(Z_test[:, i], Z_pred[:, i]) for i in range(Z_test.shape[1])]
    mae = [mean_absolute_error(Z_test[:, i], Z_pred[:, i]) for i in range(Z_test.shape[1])]
    print("R²:", [round(v, 3) for v in r2], "Mean R²:", round(float(np.mean(r2)), 3), "Mean MAE:", round(float(np.mean(mae)), 4))
    return Z_pred

z_reg = train_z_regressor(X_diff_train_s, Z_train)
Z_pred_test = evaluate_z_regressor(z_reg, X_diff_test_s, Z_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(Z_train, y_train_win.values)

y_hat_true = clf.predict(Z_test)
p_true = clf.predict_proba(Z_test)[:, 1]
acc_true = accuracy_score(y_test_win.values, y_hat_true)
auc_true = roc_auc_score(y_test_win.values, p_true)
print(f"[Upper] Acc={acc_true:.3f} AUC={auc_true:.3f}")

y_hat_pred = clf.predict(Z_pred_test)
p_pred = clf.predict_proba(Z_pred_test)[:, 1]
acc_pred = accuracy_score(y_test_win.values, y_hat_pred)
auc_pred = roc_auc_score(y_test_win.values, p_pred)
print(f"[Pipeline] Acc={acc_pred:.3f} AUC={auc_pred:.3f}")
