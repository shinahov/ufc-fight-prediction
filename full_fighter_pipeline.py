import re
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Load & sort ---
df = pd.read_csv("fights_new.csv")
df_stats = pd.read_csv("fighter_stats_imputed.csv")
df["event_date"] = pd.to_datetime(df["event_date"])
df = df.sort_values("event_date", ascending=True).reset_index(drop=True)

# --- Helpers / Regex ---
SIDE_PREFIXES = ("a_", "b_")
PCT_RE = re.compile(r"_pct$")                    # Prozent-Metriken ausschließen
ROUND_RE = re.compile(r"^r([1-5])_")             # r1_, r2_, ... r5_
META_COLS = {
    "event","event_date","fight_title",
    "fighter_a","fighter_b","winner",
    "method","end_round","end_time","referee","time_format","judges_details"
}

def normalize_name(s):
    if isinstance(s, str):
        return s.strip().lower().replace("-", " ").replace(".", "").replace("'", "")
    return s


def compute_total_time(end_round, end_time_str):
    """
    Berechnet die vergangene Zeit im Kampf in Sekunden.
    end_round: int (1–5)
    end_time_str: 'M:SS', z.B. '4:22'
    """
    try:
        minutes, seconds = map(int, str(end_time_str).split(":"))
    except Exception:
        minutes, seconds = 0, 0
    return (int(end_round) - 1) * 5 * 60 + minutes * 60 + seconds


def categorize_method(method_raw: str) -> str:
    """
    Normalisiert die Finish-/Methodenbeschreibung in feste Kategorien.
    """
    s = (method_raw or "").strip().lower()
    s = s.replace("’", "'").replace("—", "-")

    if "overturned" in s:
        return "overturned"
    if "could not continue" in s:
        return "could_not_continue"
    if "doctor" in s:
        return "doctor_stoppage"
    if "disqualification" in s or "dq" in s:
        return "dq"
    if "submission" in s or "sub" in s:
        return "sub"
    if "ko" in s or "tko" in s:
        return "ko_tko"
    if "unanimous" in s:
        return "decision_unanimous"
    if "split" in s:
        return "decision_split"
    if "majority" in s:
        return "decision_majority"
    if "decision" in s or "dec" in s:
        return "decision_unanimous"
    return "unknown"


def compute_finish_units(method_cat: str, t: float, Ttot: float) -> float:
    """
    Berechnet die Strike-äquivalenten Finish-Punkte basierend auf Methode und Zeit.
    """
    BASE_UNITS = {
        "ko_tko": 25.0,
        "sub": 20.0,
        "doctor_stoppage": 5.0,
        "could_not_continue": 3.0,
        "dq": 1.0,
        "decision_unanimous": 1.0,
        "decision_split": 1.0,
        "decision_majority": 1.0,
        "overturned": 0.0,
        "unknown": 0.0,
    }
    TIME_AFFECTS = {"ko_tko", "sub", "doctor_stoppage", "could_not_continue", "dq"}

    base = BASE_UNITS.get(method_cat, 0.0)
    EARLY_BOOST = 2.5

    if base <= 0:
        return 0.0

    progress = max(0.0, min(1.0, t / Ttot))
    if method_cat in TIME_AFFECTS:
        return base * (1.0 + EARLY_BOOST * (1.0 - progress))
    return base


def compute_dominance_simple(m):
    """
    Berechnet den Dominanz-Score für Fighter A.
    """
    # --- Äquivalenzen ---
    TD_EQ_SIG   = 5.0
    CTRL_SIG_S  = TD_EQ_SIG / 40
    KD_EQ_SIG   = 8.0

    # --- Deltas ---
    sig_net  = float(m.get("sig_strikes_a", 0)) - float(m.get("sig_strikes_b", 0))
    td_net   = float(m.get("takedowns_a", 0))   - float(m.get("takedowns_b", 0))
    ctrl_net = float(m.get("control_sec_a", 0)) - float(m.get("control_sec_b", 0))
    kd_net   = float(m.get("knockdowns_a", 0))  - float(m.get("knockdowns_b", 0))
    winner_side = m.get("winner", None)

    # --- Basisscore in Strike-Einheiten ---
    sig_units = sig_net + TD_EQ_SIG*td_net + CTRL_SIG_S*ctrl_net + KD_EQ_SIG*kd_net

    # --- Zeitberechnung ---
    t = compute_total_time(m.get("end_round", 5), m.get("end_time", "5:00"))
    Ttot = 1500.0

    # --- Methode & Finish-Einheiten ---
    method_cat = categorize_method(m.get("method") or m.get("finish"))
    finish_units = compute_finish_units(method_cat, t, Ttot)

    # --- Gesamtscore ---
    if winner_side == "A":
        score = sig_units + finish_units
    elif winner_side == "B":
        score = sig_units - finish_units

    return {
        "score": score,
        "components": {
            "sig_units_base": sig_units,
            "finish_units": finish_units,
            "method_cat": method_cat,
            "t": t, "Ttot": Ttot,
            "sig_net": sig_net, "td_net": td_net,
            "ctrl_sec_net": ctrl_net, "kd_net": kd_net,
        },
    }



def update_elo(elo_a: float, elo_b: float, winner: str | None, fighterA: str, fighterB: str, dominance_score: float | None = None, K: float = 80, scale: float = 250.0) -> tuple[float, float]:
    """
    Berechnet neue Elo-Werte für Fighter A und B.
    Gibt (elo_a_post, elo_b_post) zurück.
    """
    # Erwartete Gewinnwahrscheinlichkeiten
    Ea = 1 / (1 + 10 ** ((elo_b - elo_a) / 250))
    Eb = 1 - Ea

    # Tatsächliche Ergebnisse
    if winner == fighterA:
        Sa, Sb = 1, 0
        sign = 1
    elif winner == fighterB:
        Sa, Sb = 0, 1
        sign = -1
    else:
        Sa, Sb = 0.5, 0.5  # Draw oder NC
        sign = 0

    dom_mult = 1.0
    if dominance_score is not None:

        adj = sign * dominance_score / scale
        adj = max(-1.0, min(1.0, adj))  # clamp
        dom_mult = 1 + adj

    K *= dom_mult

    # Neue Ratings
    ra_post = elo_a + K * (Sa - Ea)
    rb_post = elo_b + K * (Sb - Eb)

    return ra_post, rb_post

def new_fighter_state():
    """Cumulative store per fighter: dynamic numeric metrics + fights/wins."""
    d = defaultdict(float)
    d["fights"] = 0.0
    d["wins"]   = 0.0
    d["elo"] = 1500.0
    return d

cum = defaultdict(new_fighter_state)
df_stats = df_stats[~df_stats["name"].duplicated(keep=False)]
df_stats["name"] = df_stats["name"].apply(normalize_name)
stats_index = df_stats.set_index("name")

def get_stats(name : str, on_date : pd.Timestamp) -> dict:
    STANCE_MAP = {
        "orthodox": 0,
        "southpaw": 1,
        "switch": 2
    }

    key = normalize_name(name)
    if key not in stats_index.index:
        return {}
    row = stats_index.loc[key]

    out = {}
    out["height"] = float(row["height"]) if pd.notna(row["height"]) else np.nan
    out["weight"] = float(row["weight"]) if pd.notna(row["weight"]) else np.nan
    out["reach"]  = float(row["reach"])  if pd.notna(row["reach"])  else np.nan
    # age are from 2024
    out["age"]    = row["age"]+1 - (2025 - on_date.year) if pd.notna(row["age"]) else np.nan
    if pd.notna(row["stance"]):
        stance = str(row["stance"]).strip().lower()
        out["stance"] = STANCE_MAP.get(stance, -1)
    else:
        out["stance"] = -1

    return out

    #if "height" in row and pd.notna

def is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

def extract_additions(row: pd.Series, side_prefix: str) -> dict[str, float]:
    """
    Liefert ALLE additiven Metriken für eine Seite:
    - inkludiert round-based: rX_<side>..., key -> 'rX_<metricohneside>'
    - inkludiert non-round: <side>..., key -> '<metricohneside>'
    - exkludiert *_pct
    - nur numerische Werte
    """
    add = defaultdict(float)

    # 1) Round-based rX_<side>...
    for col, val in row.items():
        if not is_number(val):
            continue
        m = ROUND_RE.match(col)
        if not m:
            continue
        # nach rX_ muss side_prefix folgen
        rest = col[m.end():]
        if not rest.startswith(side_prefix):
            continue
        if PCT_RE.search(rest):  # rX_a_ss_pct etc. raus
            continue
        metric_no_side = rest[len(side_prefix):]  # z.B. 'sig_l', 'td_a', 'ctrl_sec'
        key = f"r{m.group(1)}_{metric_no_side}"   # z.B. 'r1_sig_l'
        add[key] += float(val)

    # 2) Non-round <side>...
    for col, val in row.items():
        if not is_number(val):
            continue
        if not col.startswith(side_prefix):
            continue
        if PCT_RE.search(col):
            continue
        if ROUND_RE.match(col):  # rX_* schon oben behandelt
            continue
        metric_no_side = col[len(side_prefix):]   # z.B. 'sig_landed', 'sig_l', ...
        add[metric_no_side] += float(val)

    return dict(add)

# --------------- Main iteration ---------------
snap_rows = []

for i, z in df.iterrows():
    fighterA, fighterB = z["fighter_a"], z["fighter_b"]
    date = z["event_date"]

    # prior states (copy to avoid mutation later)
    priorA = dict(cum[fighterA])
    priorB = dict(cum[fighterB])
    prior_stats_A = get_stats(fighterA, date)
    prior_stats_B = get_stats(fighterB, date)

    # ---- Write snapshots (prior) ----
    # We attach ALL raw fight columns + prior__*
    raw = z.to_dict()

    snap_rows.append({
        "event_date": date,
        "fight_order": i,
        "fighter": fighterA, "side": "A",
        "opponent": fighterB,
        "winner": z["winner"],
        **raw,
        **{f"prior__{k}": float(v) for k, v in priorA.items()},
        **{f"prior_stats__{k}": float(v) if is_number(v) else v for k, v in prior_stats_A.items()}
    })
    snap_rows.append({
        "event_date": date,
        "fight_order": i,
        "fighter": fighterB, "side": "B",
        "opponent": fighterA,
        "winner": z["winner"],
        **raw,
        **{f"prior__{k}": float(v) for k, v in priorB.items()},
        **{f"prior_stats__{k}": float(v) if is_number(v) else v for k, v in prior_stats_B.items()}
    })

    elo_a = cum[fighterA]["elo"]
    elo_b = cum[fighterB]["elo"]
    winner_side = "A" if z["winner"] == fighterA else "B"

    m = {
        "sig_strikes_a": z.get("a_sig_landed", 0),
        "sig_strikes_b": z.get("b_sig_landed", 0),
        "takedowns_a": z.get("a_td_landed", 0),
        "takedowns_b": z.get("b_td_landed", 0),
        "control_sec_a": z.get("a_ctrl_sec", 0),
        "control_sec_b": z.get("b_ctrl_sec", 0),
        "knockdowns_a": z.get("a_kd", 0),
        "knockdowns_b": z.get("b_kd", 0),
        "method": z.get("method", ""),
        "end_round": z.get("end_round", 3),
        "end_time": z.get("end_time", "5:00"),
        "winner": winner_side
    }

    dom_res = compute_dominance_simple(m)
    print(f"Fight {i}: {fighterA} vs {fighterB}, Dom Score A: {dom_res['score']:.2f}")
    print(f"m details: {m}")
    dom_score = dom_res["score"]
    ra_post, rb_post = update_elo(elo_a, elo_b, z["winner"], fighterA, fighterB, dominance_score=dom_score, K=120)

    # ---- Update cumulative states with this fight ----
    # fights/wins
    cum[fighterA]["fights"] += 1
    cum[fighterB]["fights"] += 1
    if z["winner"] == fighterA:
        cum[fighterA]["wins"] += 1
    elif z["winner"] == fighterB:
        cum[fighterB]["wins"] += 1
    # draws/NC: keine wins-Erhöhung

    # numeric additions for A and B (incl. rX_*, excl. *_pct)
    addA = extract_additions(z, "a_")
    addB = extract_additions(z, "b_")
    for k, v in addA.items():
        cum[fighterA][k] += v
        cum[fighterB][k + "_rec"] += v
    for k, v in addB.items():
        cum[fighterB][k] += v
        cum[fighterA][k + "_rec"] += v

    cum[fighterA]["elo"] = ra_post
    cum[fighterB]["elo"] = rb_post

# --------------- Result DataFrames ---------------
snap = pd.DataFrame(snap_rows).sort_values(
    ["event_date", "fight_order", "fighter"]
).reset_index(drop=True)


def compute_derived_features(snap):
    snap["prior_feat__sig_acc"] = np.where(
        (snap["prior__sig_attempts"] > 0) & (snap["prior__fights"] > 0),
        snap["prior__sig_landed"] / snap["prior__sig_attempts"],
        np.nan
    )
    return snap


snap = compute_derived_features(snap)


# Convenience: history getter
def get_fighter_history(name: str) -> pd.DataFrame:
    """Alle Zeilen für 'name' mit Rohdaten + prior__* (inkl. rundenbasierter prior__rX_*)."""
    out = snap.loc[snap["fighter"] == name].copy()
    return out.sort_values(["event_date", "fight_order"]).reset_index(drop=True)

# Optional: nur prior-Spalten (breit)
def fighter_prior_matrix(name: str) -> pd.DataFrame:
    """Wide-Matrix der prior-Werte (nur prior__*), Index = Fight-Order/Date."""
    hist = get_fighter_history(name)
    prior_cols = [c for c in hist.columns if c.startswith("prior__")]
    base = hist[["event_date", "fight_order", "fighter", "opponent", "winner"]]
    return pd.concat([base, hist[prior_cols]], axis=1)

pd.set_option("display.max_columns", None)   # zeigt ALLE Spalten
pd.set_option("display.width", None)         # kein Zeilenumbruch
pd.set_option("display.max_colwidth", None)  # volle Spaltenbreite
pd.set_option("display.max_rows", None)      # alle Zeilen (vorsichtig bei großen DFs!)

def side_prefix_from(side: str) -> str:
    return "a_" if side == "A" else "b_"

def extract_prior_dict(row: pd.Series) -> dict[str, float]:
    """
    prior__*-Werte in ein dict ohne Prefix mappen. Fehlende/NaN -> 0.
    """
    out = {}
    for col, val in row.items():
        if isinstance(col, str) and col.startswith("prior__"):
            k = col[len("prior__"):]
            v = 0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
            out[k] = v
    return out

def validate_snap(snap: pd.DataFrame, atol: float = 1e-6) -> pd.DataFrame:
    """
    Prüft für alle Fighter die Rekonstruktion der prior-Zustände.
    Gibt ein DataFrame mit Abweichungen zurück (leer = alles korrekt).
    """
    errors = []

    for fighter in snap["fighter"].dropna().unique():
        hist = (
            snap.loc[snap["fighter"] == fighter]
                .sort_values(["event_date", "fight_order"], kind="stable")
                .reset_index(drop=True)
        )
        if hist.empty:
            continue

        # running_total = prior des ersten Kampfes (als dict)
        running = extract_prior_dict(hist.iloc[0])

        # Iteriere ab dem zweiten Eintrag; vergleiche prior[i] mit running nach Update von i-1
        for i in range(1, len(hist)):
            prev = hist.iloc[i-1]
            curr = hist.iloc[i]

            # 1) Update fights/wins basierend auf prev
            running["fights"] = running.get("fights", 0.0) + 1.0
            if prev.get("winner", None) == prev.get("fighter", None):
                running["wins"] = running.get("wins", 0.0) + 1.0

            # 2) Addiere alle numerischen Fight-Additions (inkl. rX_*), exkl. *_pct
            side_prefix = side_prefix_from(prev["side"])
            adds = extract_additions(prev, side_prefix)
            for k, v in adds.items():
                running[k] = running.get(k, 0.0) + float(v)

            # 3) Erwartung vs. tatsächliches prior im aktuellen Datensatz
            curr_prior = extract_prior_dict(curr)

            # vereinheitliche Keys (alles was vorkommt)
            keys = set(running.keys()) | set(curr_prior.keys())

            for k in keys:
                if k == "elo":
                    continue
                exp = running.get(k, 0.0)
                got = curr_prior.get(k, 0.0)
                if not np.isclose(exp, got, atol=atol, rtol=0):
                    errors.append({
                        "fighter": fighter,
                        "row_index": int(curr.name),
                        "fight_order": int(curr["fight_order"]),
                        "event_date": curr["event_date"],
                        "metric": k,
                        "expected_prior": exp,
                        "got_prior": got,
                        "delta": got - exp
                    })

    return pd.DataFrame(errors)




err_df = validate_snap(snap, atol=1e-6)

if err_df.empty:
    print("OK: Alle prior-Zustände sind konsistent rekonstruiert.")
else:
    print(f"FEHLER: {len(err_df)} Abweichungen gefunden.")
    # Zeige ein paar Beispiele
    print(err_df.sort_values(["fighter", "event_date", "fight_order"]).head(20))
    # Optional: alle Abweichungen speichern
    err_df.to_csv("prior_validation_errors.csv", index=False)

#pd.set_option("display.max_rows", None)        # zeigt ALLE Zeilen
#pd.set_option("display.max_columns", None)     # zeigt ALLE Spalten
#pd.set_option("display.width", None)           # keine Zeilenumbrüche
#pd.set_option("display.max_colwidth", None)    # volle Spaltenbreite
print(snap.tail(10))
snap.to_csv("snapshot.csv", index=False)

snap2 = pd.read_csv("snapshot.csv")
print(len(snap2), "Zeilen")       # sollte == len(snap)
print(len(snap2.columns), "Spalten")

final_elos = [(name, stats["elo"]) for name, stats in cum.items()]
top20 = sorted(final_elos, key=lambda x: x[1], reverse=True)[:20]

print("Top 20 Fighter nach Elo:")
for name, elo in top20:
    print(f"{name:25s} {elo:.2f}")

#print(snap)
