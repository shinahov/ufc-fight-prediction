# calculate features
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Optional

from preprocessing import PCT_PATTERN, ROUND_PATTERN, is_numeric
from Config import STANCE_MAP


def extract_fight_additions(
        row: pd.Series,
        side_prefix: str) -> Dict[str, float]:
    additions = defaultdict(float)
    # extract additional fight stats for a fighter side (A or B)
    for col, val in row.items():
        if not is_numeric(val):
            continue

        # Round-based: r1_a_sig_l -> r1_sig_l
        round_match = ROUND_PATTERN.match(col)
        if round_match:
            rest = col[round_match.end():]
            if not rest.startswith(side_prefix):
                continue
            if PCT_PATTERN.search(rest):
                continue
            metric = rest[len(side_prefix):]
            key = f"r{round_match.group(1)}_{metric}"
            additions[key] += float(val)
            continue

        # Non-round: a_sig_landed -> sig_landed
        if not col.startswith(side_prefix):
            continue
        if PCT_PATTERN.search(col):
            continue

        metric = col[len(side_prefix):]
        additions[metric] += float(val)

    return dict(additions)


def get_fighter_static_stats(
        fighter_name: str,
        stats_index: pd.DataFrame,
        fight_date: pd.Timestamp) -> Dict[str, float]:
    # get all static stats for a fighter at the time of the fight
    from preprocessing import normalize_name

    key = normalize_name(fighter_name)
    if key not in stats_index.index:
        return {}

    row = stats_index.loc[key]

    out = {}
    out["height"] = float(row["height"]) if pd.notna(row["height"]) else np.nan
    out["weight"] = float(row["weight"]) if pd.notna(row["weight"]) else np.nan
    out["reach"] = float(row["reach"]) if pd.notna(row["reach"]) else np.nan

    # data is from 2024, adjust age accordingly
    if pd.notna(row["age"]):
        out["age"] = row["age"] + 1 - (2025 - fight_date.year)
    else:
        out["age"] = np.nan

    # Stance encoding
    if pd.notna(row["stance"]):
        stance = str(row["stance"]).strip().lower()
        out["stance"] = STANCE_MAP.get(stance, -1)
    else:
        out["stance"] = -1

    return out


def compute_derived_features(snap: pd.DataFrame, eps=1e-9) -> pd.DataFrame:
    # compute derived features based on prior stats
    df = snap.copy()

    # Ensure datetime
    if not np.issubdtype(df["event_date"].dtype, np.datetime64):
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    df = df.sort_values(
        ["event_date", "fight_order", "fighter"],
        kind="mergesort"
    ).reset_index(drop=True)


    fights = df["prior__fights"].fillna(0.0).astype(float)
    valid = fights > 0

    t_sec = df["prior__time_in_fight"].fillna(0.0).astype(float)
    mins = np.maximum(t_sec / 60.0, eps)
    per15 = 15.0 / np.maximum(mins, eps)

    def safe_div(a, b):
        return a.astype(float) / np.maximum(b.astype(float), eps)

    # accuracy metrics
    df["prior__sig_str_acc"] = np.where(
        valid,
        safe_div(df["prior__sig_landed"], df["prior__sig_attempts"]),
        np.nan
    )
    df["prior__td_acc"] = np.where(
        valid,
        safe_div(df["prior__td_landed"], df["prior__td_attempts"]),
        np.nan
    )

    # def avoidance metrics
    df["prior__sig_str_def"] = np.where(
        valid,
        1.0 - safe_div(df["prior__sig_landed_rec"], df["prior__sig_attempts_rec"]),
        np.nan
    )
    df["prior__td_def"] = np.where(
        valid,
        1.0 - safe_div(df["prior__td_landed_rec"], df["prior__td_attempts_rec"]),
        np.nan
    )

    # per-minute rates
    df["prior__sig_att_per_min"] = np.where(valid, df["prior__sig_attempts"] / mins, np.nan)
    df["prior__sig_land_per_min"] = np.where(valid, df["prior__sig_landed"] / mins, np.nan)
    df["prior__td_att_per_min"] = np.where(valid, df["prior__td_attempts"] / mins, np.nan)
    df["prior__td_land_per_min"] = np.where(valid, df["prior__td_landed"] / mins, np.nan)
    df["prior__ctrl_per_min"] = np.where(
        valid,
        df["prior__ctrl_sec"] / np.maximum(t_sec, eps) * 60.0,
        np.nan
    )

    # per 15-min rates
    df["prior__sub_att_per15"] = np.where(valid, df["prior__sub_att"] * per15, np.nan)
    df["prior__rev_per15"] = np.where(valid, df["prior__rev"] * per15, np.nan)
    df["prior__kd_per15"] = np.where(valid, df["prior__kd"] * per15, np.nan)
    df["prior__kd_against_per15"] = np.where(valid, df["prior__kd_rec"] * per15, np.nan)

    # strike location distribution
    ss_attempts = np.maximum(df["prior__ss_attempts"].fillna(0.0), eps)
    df["prior__share_head_attempts"] = np.where(
        valid, df["prior__ss_head_a"].fillna(0) / ss_attempts, np.nan
    )
    df["prior__share_body_attempts"] = np.where(
        valid, df["prior__ss_body_a"].fillna(0) / ss_attempts, np.nan
    )
    df["prior__share_leg_attempts"] = np.where(
        valid, df["prior__ss_leg_a"].fillna(0) / ss_attempts, np.nan
    )

    # position distribution
    df["prior__share_distance_attempts"] = np.where(
        valid, df["prior__ss_distance_a"].fillna(0) / ss_attempts, np.nan
    )
    df["prior__share_clinch_attempts"] = np.where(
        valid, df["prior__ss_clinch_a"].fillna(0) / ss_attempts, np.nan
    )
    df["prior__share_ground_attempts"] = np.where(
        valid, df["prior__ss_ground_a"].fillna(0) / ss_attempts, np.nan
    )

    # margin metrics
    df["prior__sig_land_margin_per_min"] = np.where(
        valid,
        (df["prior__sig_landed"] - df["prior__sig_landed_rec"]) / mins,
        np.nan
    )
    df["prior__td_land_margin_per_min"] = np.where(
        valid,
        (df["prior__td_landed"] - df["prior__td_landed_rec"]) / mins,
        np.nan
    )
    df["prior__ctrl_margin_per_min"] = np.where(
        valid,
        (df["prior__ctrl_sec"] - df["prior__ctrl_sec_rec"]) / np.maximum(t_sec, eps) * 60.0,
        np.nan
    )

    # other metrics
    df["prior__winrate"] = np.where(valid, safe_div(df["prior__wins"], fights), np.nan)
    df["prior__avg_fight_len_min"] = np.where(valid, (t_sec / fights) / 60.0, np.nan)

    return df