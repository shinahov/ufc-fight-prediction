import re
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Load & sort ---
df = pd.read_csv("fights_new.csv")
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
def update_elo(elo_a: float, elo_b: float, winner: str | None, fighterA: str, fighterB: str, K: float = 32):
    """
    Berechnet neue Elo-Werte für Fighter A und B.
    Gibt (elo_a_post, elo_b_post) zurück.
    """
    # Erwartete Gewinnwahrscheinlichkeiten
    Ea = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    Eb = 1 - Ea

    # Tatsächliche Ergebnisse
    if winner == fighterA:
        Sa, Sb = 1, 0
    elif winner == fighterB:
        Sa, Sb = 0, 1
    else:
        Sa, Sb = 0.5, 0.5  # Draw oder NC

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
        **{f"prior__{k}": float(v) for k, v in priorA.items()}
    })
    snap_rows.append({
        "event_date": date,
        "fight_order": i,
        "fighter": fighterB, "side": "B",
        "opponent": fighterA,
        "winner": z["winner"],
        **raw,
        **{f"prior__{k}": float(v) for k, v in priorB.items()}
    })

    elo_a = cum[fighterA]["elo"]
    elo_b = cum[fighterB]["elo"]
    ra_post, rb_post = update_elo(elo_a, elo_b, z["winner"], fighterA, fighterB)

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
    for k, v in addB.items():
        cum[fighterB][k] += v

    cum[fighterA]["elo"] = ra_post
    cum[fighterB]["elo"] = rb_post

# --------------- Result DataFrames ---------------
snap = pd.DataFrame(snap_rows).sort_values(
    ["event_date", "fight_order", "fighter"]
).reset_index(drop=True)

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
