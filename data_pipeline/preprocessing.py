import re
import numpy as np
import pandas as pd
from typing import Optional

# Regex Patterns
PCT_PATTERN = re.compile(r"_pct$")
ROUND_PATTERN = re.compile(r"^r([1-5])_")

def is_numeric(value) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
    )


def normalize_name(name: Optional[str]) -> Optional[str]:
    # normalize fighter names for consistent matching
    if not isinstance(name, str):
        return name
    return (
        name.strip()
        .lower()
        .replace("-", " ")
        .replace(".", "")
        .replace("'", "")
    )


def parse_mmss(time_str: Optional[str]) -> int:
    # Convert "MM:SS" formatted string to total seconds
    if time_str is None:
        return 0
    if isinstance(time_str, float) and np.isnan(time_str):
        return 0

    time_str = str(time_str).strip()
    if not time_str or ":" not in time_str:
        return 0

    try:
        parts = time_str.split(":")
        minutes, seconds = int(parts[0]), int(parts[1])
        return minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0


def compute_fight_seconds(
        end_round: int,
        end_time: str,
        round_length: int = 300) -> int:    # Compute total fight duration in seconds
    try:
        round_num = int(end_round)
    except (ValueError, TypeError):
        round_num = 1

    elapsed_in_round = parse_mmss(end_time)
    return max(0, (round_num - 1) * round_length + elapsed_in_round)


def categorize_method(method_raw: Optional[str]) -> str:
    # categorize fight outcome method into predefined categories
    if not method_raw:
        return "unknown"

    s = method_raw.strip().lower()
    s = s.replace("'", "'").replace("—", "-")

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

def load_fights(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date", ascending=True).reset_index(drop=True)
    return df

def load_fighter_stats(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[~df["name"].duplicated(keep=False)]
    df["name"] = df["name"].apply(normalize_name)
    return df