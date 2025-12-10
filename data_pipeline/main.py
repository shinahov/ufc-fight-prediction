from typing import Tuple

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from pandas import DataFrame

from Config import EloConfig, DATA_DIR, OUTPUT_DIR, FIGHTS_FILE, STATS_FILE, SNAPSHOT_FILE
from data_pipeline.elo import EloSystem
from preprocessing import load_fights, load_fighter_stats, normalize_name, compute_fight_seconds

from features import (
    extract_fight_additions,
    get_fighter_static_stats,
    compute_derived_features
)
from elo import EloSystem


def create_fighter_state() -> dict:
    state = defaultdict(float)
    state["fights"] = 0.0
    state["wins"] = 0.0
    return state


def build_snapshot_row(
        fight_row: pd.Series,
        fighter: str,
        opponent: str,
        side: str,
        fight_order: int,
        prior_state: dict,
        prior_stats: dict,
        last_fight_days: float) -> dict:
    row = {
        "event_date": fight_row["event_date"],
        "fight_order": fight_order,
        "fighter": fighter,
        "side": side,
        "opponent": opponent,
        "winner": fight_row["winner"],
        **fight_row.to_dict(),
    }

    # Prior cumulative stats
    for k, v in prior_state.items():
        if isinstance(v, (int, float, np.integer, np.floating)) or pd.isna(v):
            row[f"prior__{k}"] = float(v) if not pd.isna(v) else np.nan

    row["prior__last_fight_days"] = last_fight_days

    # Prior static stats
    for k, v in prior_stats.items():
        if isinstance(v, (int, float)):
            row[f"prior_stats__{k}"] = v
        else:
            row[f"prior_stats__{k}"] = v

    return row


def process_fights(
        fights_df: pd.DataFrame,
        stats_df: pd.DataFrame) -> tuple[DataFrame, EloSystem]:
    # Setup
    elo_system = EloSystem(EloConfig())
    cumulative = defaultdict(create_fighter_state)
    stats_index = stats_df.set_index("name")

    snapshot_rows = []

    for fight_idx, fight in fights_df.iterrows():
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]
        event_date = fight["event_date"]

        # get prior cumulative stats
        prior_a = dict(cumulative[fighter_a])
        prior_b = dict(cumulative[fighter_b])

        prior_a["elo"] = elo_system.get_rating(fighter_a)
        prior_b["elo"] = elo_system.get_rating(fighter_b)

        # Static stats
        stats_a = get_fighter_static_stats(fighter_a, stats_index, event_date)
        stats_b = get_fighter_static_stats(fighter_b, stats_index, event_date)

        # Days since last fight
        last_date_a = cumulative[fighter_a].get("last_fight_date")
        last_date_b = cumulative[fighter_b].get("last_fight_date")
        days_a = float((event_date - last_date_a).days) if last_date_a else np.nan
        days_b = float((event_date - last_date_b).days) if last_date_b else np.nan

        # Build snapshot rows
        snapshot_rows.append(build_snapshot_row(
            fight, fighter_a, fighter_b, "A", fight_idx,
            prior_a, stats_a, days_a
        ))
        snapshot_rows.append(build_snapshot_row(
            fight, fighter_b, fighter_a, "B", fight_idx,
            prior_b, stats_b, days_b
        ))


        # Fights & wins
        cumulative[fighter_a]["fights"] += 1
        cumulative[fighter_b]["fights"] += 1

        if fight["winner"] == fighter_a:
            cumulative[fighter_a]["wins"] += 1
        elif fight["winner"] == fighter_b:
            cumulative[fighter_b]["wins"] += 1

        # Fight time
        fight_seconds = compute_fight_seconds(
            fight.get("end_round"),
            fight.get("end_time")
        )
        cumulative[fighter_a]["time_in_fight"] += fight_seconds
        cumulative[fighter_b]["time_in_fight"] += fight_seconds

        # Fight stats additions
        adds_a = extract_fight_additions(fight, "a_")
        adds_b = extract_fight_additions(fight, "b_")

        for k, v in adds_a.items():
            cumulative[fighter_a][k] += v
            cumulative[fighter_b][k + "_rec"] += v

        for k, v in adds_b.items():
            cumulative[fighter_b][k] += v
            cumulative[fighter_a][k + "_rec"] += v

        # Elo update
        winner_side = "A" if fight["winner"] == fighter_a else "B"

        dom_result = elo_system.dominance_calc.compute(
            sig_a=fight.get("a_sig_landed", 0),
            sig_b=fight.get("b_sig_landed", 0),
            td_a=fight.get("a_td_landed", 0),
            td_b=fight.get("b_td_landed", 0),
            ctrl_a=fight.get("a_ctrl_sec", 0),
            ctrl_b=fight.get("b_ctrl_sec", 0),
            kd_a=fight.get("a_kd", 0),
            kd_b=fight.get("b_kd", 0),
            winner_side=winner_side,
            method=fight.get("method", ""),
            end_round=fight.get("end_round", 3),
            end_time=fight.get("end_time", "5:00")
        )

        elo_system.update(
            fighter_a, fighter_b,
            fight["winner"],
            dominance_score=dom_result.score
        )

        # Update last fight date
        cumulative[fighter_a]["last_fight_date"] = event_date
        cumulative[fighter_b]["last_fight_date"] = event_date

    # Build DataFrame
    snapshot = pd.DataFrame(snapshot_rows)
    snapshot = snapshot.sort_values(
        ["event_date", "fight_order", "fighter"]
    ).reset_index(drop=True)

    return snapshot, elo_system


def main():
    print("Loading data...")
    fights_df = load_fights(f"{DATA_DIR}/{FIGHTS_FILE}")
    stats_df = load_fighter_stats(f"{DATA_DIR}/{STATS_FILE}")

    print(f"Processing {len(fights_df)} fights...")
    snapshot, elo_system = process_fights(fights_df, stats_df)

    print("Computing derived features...")
    snapshot = compute_derived_features(snapshot)

    # Save
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    output_path = f"{OUTPUT_DIR}/{SNAPSHOT_FILE}"
    snapshot.to_csv(output_path, index=False)
    print(f"Saved snapshot to {output_path}")
    print(f"  {len(snapshot)} rows, {len(snapshot.columns)} columns")

    # Top fighters
    print("\nTop 20 Fighters by Elo:")
    for name, rating in elo_system.get_top_fighters(20):
        print(f"  {name:25s} {rating:.2f}")


if __name__ == "__main__":
    main()