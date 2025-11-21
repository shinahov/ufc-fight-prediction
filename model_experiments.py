import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def build_dataset_from_snapshot(df):
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Filters only fighters with prior fights + known winner
    df = df[(df["prior__fights"] > 0) & (df["winner"].notna())].copy()
    grp = df.groupby("fight_order")["side"].nunique()
    valid_ids = grp[grp == 2].index
    df = df[df["fight_order"].isin(valid_ids)].copy()

    cnt = df.groupby("fight_order").size()
    valid_fights = cnt[cnt == 2].index
    df = df[df["fight_order"].isin(valid_fights)].copy()

    # A/B Split & Sort
    A = df[df["side"] == "A"].sort_values("fight_order").reset_index(drop=True)
    B = df[df["side"] == "B"].sort_values("fight_order").reset_index(drop=True)

    assert np.array_equal(
        A["fight_order"].values, B["fight_order"].values
    ), "A/B nicht ausrichtbar."


    y = (A["winner"] == A["fighter"]).astype(np.float32).values.reshape(-1, 1)
    print("Label-Verteilung (Anteil 1):", float(y.mean()))


    num_cols = df.select_dtypes(include=[np.number]).columns

    # Priors
    prior_cols = [
        c for c in num_cols
        if c.startswith("prior_") or c.startswith("prior_stats__")
    ]

    prior_cols = [c for c in prior_cols if "stance" not in c]


    fight_stat_prefixes = ["a_", "b_", "r1_", "r2_", "r3_", "r4_", "r5_"]
    fight_cols = [
        c for c in num_cols
        if any(c.startswith(p) for p in fight_stat_prefixes)
    ]

    # Priors A/B → diff + ratio
    A_pr = A[prior_cols].to_numpy(dtype=np.float32)
    B_pr = B[prior_cols].to_numpy(dtype=np.float32)

    X_diff = A_pr - B_pr
    eps = 1e-6
    X_ratio = (A_pr + 1.0) / (B_pr + 1.0 + eps)

    X = np.hstack([X_diff, X_ratio]).astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    X_cols = [f"diff::{c}" for c in prior_cols] + [f"ratio::{c}" for c in prior_cols]


    S = A[fight_cols].fillna(0.0).to_numpy(dtype=np.float32)
    S_cols = list(fight_cols)
    S_dim = S.shape[1]

    # Splits (80% Train, 10% Val, 10% Test)
    n = len(X)
    train_size = 0.80
    val_size = 0.10

    tr_end = int(train_size * n)
    va_end = int((train_size + val_size) * n)

    X_tr, X_va, X_te = X[:tr_end], X[tr_end:va_end], X[va_end:]
    S_tr, S_va, S_te = S[:tr_end], S[tr_end:va_end], S[va_end:]
    y_tr, y_va, y_te = y[:tr_end], y[tr_end:va_end], y[va_end:]

    print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

    # Skaling
    scaler_X = StandardScaler()
    scaler_S = StandardScaler()

    X_tr_s = scaler_X.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler_X.transform(X_va).astype(np.float32)
    X_te_s = scaler_X.transform(X_te).astype(np.float32)

    S_tr_s = scaler_S.fit_transform(S_tr).astype(np.float32)
    S_va_s = scaler_S.transform(S_va).astype(np.float32)
    S_te_s = scaler_S.transform(S_te).astype(np.float32)

    return {
        "X_tr_s": X_tr_s,
        "X_va_s": X_va_s,
        "X_te_s": X_te_s,
        "S_tr_s": S_tr_s,
        "S_va_s": S_va_s,
        "S_te_s": S_te_s,
        "y_tr": y_tr,
        "y_va": y_va,
        "y_te": y_te,
        "X_cols": X_cols,
        "S_cols": S_cols,
        "S_dim": S_dim,
        "scaler_X": scaler_X,
        "scaler_S": scaler_S,
    }



def train_ae_win(
    data,
    latent_dim=8,
    alpha_win=2.0,
    beta_rec=1.0,
    batch_size=256,
    max_epochs=200,
    patience=20,
):
    S_tr_s = data["S_tr_s"]
    S_va_s = data["S_va_s"]
    S_te_s = data["S_te_s"]

    # y as 1D
    y_tr = data["y_tr"][:, 0]
    y_va = data["y_va"][:, 0]
    y_te = data["y_te"][:, 0]

    S_dim = data["S_dim"]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # Input: Fight-Stats
    inp_stats = layers.Input(shape=(S_dim,), name="inp_stats")

    # Encoder
    h = layers.Dense(256, activation="relu", name="enc_1")(inp_stats)
    h = layers.Dense(128, activation="relu", name="enc_2")(h)
    z = layers.Dense(latent_dim, activation="relu", name="z_latent")(h)

    # Decoder (Reconstruction)
    d = layers.Dense(128, activation="relu", name="dec_1")(z)
    stats_out = layers.Dense(S_dim, activation="linear", name="stats_out")(d)


    c = layers.Dense(64, activation="relu", name="cls_1")(z)
    win_out = layers.Dense(1, activation="sigmoid", name="win_out")(c)

    ae_model = keras.Model(
        inputs=inp_stats,
        outputs={"stats_out": stats_out, "win_out": win_out},
        name="ae_win_model",
    )

    ae_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"stats_out": "mse", "win_out": "binary_crossentropy"},
        loss_weights={"stats_out": beta_rec, "win_out": alpha_win},
        metrics={"win_out": [keras.metrics.AUC(name="auc"), "accuracy"]},
    )

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_win_out_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    ae_model.fit(
        S_tr_s,
        {"stats_out": S_tr_s, "win_out": y_tr},
        validation_data=(S_va_s, {"stats_out": S_va_s, "win_out": y_va}),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cb,
    )


    encoder = keras.Model(
        inputs=inp_stats,
        outputs=z,
        name="fight_stats_encoder",
    )


    def eval_split(Ss, ys, name):
        out = ae_model.predict(Ss, verbose=0)
        p = out["win_out"][:, 0]
        preds = (p >= 0.5).astype(int)
        acc = accuracy_score(ys, preds)
        auc = roc_auc_score(ys, p)
        print(f"[AE] {name}: ACC={acc:.3f}  AUC={auc:.3f}")

    eval_split(S_va_s, y_va, "VAL")
    eval_split(S_te_s, y_te, "TEST")

    return {
        "ae_model": ae_model,
        "encoder": encoder,
    }



def train_prior_to_latent(
    data,
    encoder,
    latent_dim=8,
    gamma_z=1.0,      # MSE(z_hat, z_true)
    delta_win=2.0,    # Winner-Loss
    batch_size=256,
    max_epochs=200,
    patience=20,
):
    X_tr_s = data["X_tr_s"]
    X_va_s = data["X_va_s"]
    X_te_s = data["X_te_s"]
    S_tr_s = data["S_tr_s"]
    S_va_s = data["S_va_s"]
    S_te_s = data["S_te_s"]

    y_tr = data["y_tr"][:, 0]
    y_va = data["y_va"][:, 0]
    y_te = data["y_te"][:, 0]

    # Encoder: z_true = f(S)
    z_tr_true = encoder.predict(S_tr_s, verbose=0)
    z_va_true = encoder.predict(S_va_s, verbose=0)
    z_te_true = encoder.predict(S_te_s, verbose=0)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    in_dim = X_tr_s.shape[1]
    inp_priors = layers.Input(shape=(in_dim,), name="inp_priors")

    # Mapping priors → z_hat
    h = layers.Dense(256, activation="relu", name="map_1")(inp_priors)
    h = layers.Dense(128, activation="relu", name="map_2")(h)
    z_hat = layers.Dense(latent_dim, activation="linear", name="z_hat")(h)

    # Winner-Head
    c = layers.Dense(64, activation="relu", name="cls_2")(z_hat)
    win_out = layers.Dense(1, activation="sigmoid", name="win_out")(c)

    prior_model = keras.Model(
        inputs=inp_priors,
        outputs={"z_out": z_hat, "win_out": win_out},
        name="priors_to_latent_model",
    )

    prior_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"z_out": "mse", "win_out": "binary_crossentropy"},
        loss_weights={"z_out": gamma_z, "win_out": delta_win},
        metrics={"win_out": [keras.metrics.AUC(name="auc"), "accuracy"]},
    )

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_win_out_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    prior_model.fit(
        X_tr_s,
        {"z_out": z_tr_true, "win_out": y_tr},
        validation_data=(X_va_s, {"z_out": z_va_true, "win_out": y_va}),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cb,
    )

    # Evaluation
    def eval_split(Xs, ys, name):
        out = prior_model.predict(Xs, verbose=0)
        p = out["win_out"][:, 0]
        preds = (p >= 0.5).astype(int)
        acc = accuracy_score(ys, preds)
        auc = roc_auc_score(ys, p)
        print(f"[Priors→z] {name}: ACC={acc:.3f}  AUC={auc:.3f}")

    eval_split(X_va_s, y_va, "VAL")
    eval_split(X_te_s, y_te, "TEST")

    return prior_model

def train_zhat_to_win(
    data,
    prior_model,
    batch_size=256,
    max_epochs=200,
    patience=20,
):

    X_tr_s = data["X_tr_s"]
    X_va_s = data["X_va_s"]
    X_te_s = data["X_te_s"]

    y_tr = data["y_tr"][:, 0]
    y_va = data["y_va"][:, 0]
    y_te = data["y_te"][:, 0]


    z_tr_hat = prior_model.predict(X_tr_s, verbose=0)["z_out"]
    z_va_hat = prior_model.predict(X_va_s, verbose=0)["z_out"]
    z_te_hat = prior_model.predict(X_te_s, verbose=0)["z_out"]

    latent_dim = z_tr_hat.shape[1]
    print("z_hat shape:", z_tr_hat.shape)


    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    inp_z = layers.Input(shape=(latent_dim,), name="inp_z_hat")
    h = layers.Dense(32, activation="relu")(inp_z)
    win_out = layers.Dense(1, activation="sigmoid", name="win_out")(h)

    clf_model = keras.Model(inp_z, win_out, name="zhat_win_classifier")

    clf_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    clf_model.fit(
        z_tr_hat,
        y_tr,
        validation_data=(z_va_hat, y_va),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cb,
    )

    # 3) Evaluation
    def eval_split(Zs, ys, name):
        p = clf_model.predict(Zs, verbose=0)[:, 0]
        preds = (p >= 0.5).astype(int)
        acc = accuracy_score(ys, preds)
        auc = roc_auc_score(ys, p)
        print(f"[z_hat→win] {name}: ACC={acc:.3f}  AUC={auc:.3f}")
        print(f"[z_hat→win] {name}: p_mean={p.mean():.3f}, p_min={p.min():.3f}, p_max={p.max():.3f}")

    eval_split(z_va_hat, y_va, "VAL")
    eval_split(z_te_hat, y_te, "TEST")

    return clf_model

def train_classifier_only(
    data,
    batch_size=256,
    max_epochs=200,
    patience=20,
):

    X_tr_s = data["X_tr_s"]
    X_va_s = data["X_va_s"]
    X_te_s = data["X_te_s"]

    # y als 1D
    y_tr = data["y_tr"][:, 0]
    y_va = data["y_va"][:, 0]
    y_te = data["y_te"][:, 0]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    in_dim = X_tr_s.shape[1]
    inp = layers.Input(shape=(in_dim,), name="input_priors")


    win_out = layers.Dense(1, activation="sigmoid", name="win_out")(inp)

    model = keras.Model(inp, win_out, name="baseline_priors_win")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_tr_s,
        y_tr,
        validation_data=(X_va_s, y_va),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cb,
    )

    # Eval
    def eval_split(Xs, ys, name):
        p = model.predict(Xs, verbose=0)[:, 0]
        preds = (p >= 0.5).astype(int)
        acc = accuracy_score(ys, preds)
        auc = roc_auc_score(ys, p)
        print(f"[BASE] {name}: ACC={acc:.3f}  AUC={auc:.3f}")
        print(f"[BASE] {name}: p_mean={p.mean():.3f}, p_min={p.min():.3f}, p_max={p.max():.3f}")

    eval_split(X_va_s, y_va, "VAL")
    eval_split(X_te_s, y_te, "TEST")

    return model, history



if __name__ == "__main__":
    df = pd.read_csv("snapshot.csv")
    data = build_dataset_from_snapshot(df)


    print("\nPriors -> Win (Baseline Classifier)")
    base_model, base_hist = train_classifier_only(data)


    print("\nPHASE 1: AE Win")
    ae_res = train_ae_win(data, latent_dim=8)
    encoder = ae_res["encoder"]


    print("\nPHASE 2: Priors -> z_hat (+ Win)")
    prior_model = train_prior_to_latent(
        data,
        encoder=encoder,
        latent_dim=8,
    )


    Z_te_hat = prior_model.predict(data["X_te_s"], verbose=0)["z_out"]
    print("Latent-Space (Test) Shape:", Z_te_hat.shape)


    print("\nPHASE 3: z_hat -> Win (separat Classifier)")
    zhat_clf = train_zhat_to_win(data, prior_model)

