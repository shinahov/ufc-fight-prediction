from dataclasses import dataclass

STANCE_MAP = {
    "orthodox": 0,
    "southpaw": 1,
    "switch": 2,
}

METHOD_CATEGORIES = {
    "ko_tko": ["ko", "tko"],
    "sub": ["submission", "sub"],
    "dq": ["disqualification", "dq"],
    "doctor_stoppage": ["doctor"],
    "could_not_continue": ["could not continue"],
    "overturned": ["overturned"],
}

FINISH_BASE_UNITS = {
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

META_COLS = {
    "event", "event_date", "fight_title",
    "fighter_a", "fighter_b", "winner",
    "method", "end_round", "end_time",
    "referee", "time_format", "judges_details"
}

DATA_DIR = "data"
OUTPUT_DIR = "output"
FIGHTS_FILE = "fights_new.csv"
STATS_FILE = "fighter_stats_imputed.csv"
SNAPSHOT_FILE = "snapshot.csv"

@dataclass
class EloConfig:
    K: float = 120.0 # K-factor for Elo rating updates
    SCALE: float = 250.0 # Scaling factor for Elo probability calculation
    INITIAL_RATING: float = 1500.0 # Default Elo rating for new fighters


@dataclass
class DominanceWeights:
    TD_EQUIVALENT_SIG: float = 5.0      # 1 Takedown = 5 Sig Strikes
    CTRL_SEC_PER_SIG: float = 0.125     # 40 Sek Control = 5 Sig Strikes
    KD_EQUIVALENT_SIG: float = 8.0      # 1 Knockdown = 8 Sig Strikes
    EARLY_FINISH_BOOST: float = 2.5     # Early finish multiplier for final score