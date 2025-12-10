from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Tuple, Dict

from Config import EloConfig, DominanceWeights, FINISH_BASE_UNITS
from preprocessing import categorize_method, compute_fight_seconds


@dataclass
class DominanceResult:
    score: float
    sig_units: float
    finish_units: float
    method_category: str
    fight_time: int
    total_time: int


class DominanceCalculator:
# computes dominance score based on fight statistics
    def __init__(self, weights: DominanceWeights = None):
        self.weights = weights or DominanceWeights()

    def compute(
            self,
            sig_a: int, sig_b: int,
            td_a: int, td_b: int,
            ctrl_a: int, ctrl_b: int,
            kd_a: int, kd_b: int,
            winner_side: Optional[str],
            method: str,
            end_round: int,
            end_time: str,
            total_fight_time: int = 1500
    ) -> DominanceResult:
        w = self.weights

        # Netto-Diff
        sig_net = sig_a - sig_b
        td_net = td_a - td_b
        ctrl_net = ctrl_a - ctrl_b
        kd_net = kd_a - kd_b
        # sing units calculation
        sig_units = (
                sig_net
                + w.TD_EQUIVALENT_SIG * td_net
                + w.CTRL_SEC_PER_SIG * ctrl_net
                + w.KD_EQUIVALENT_SIG * kd_net
        )

        # time and finish calculation
        fight_time = compute_fight_seconds(end_round, end_time)
        method_cat = categorize_method(method)
        finish_units = self._compute_finish_units(
            method_cat, fight_time, total_fight_time
        )

        # final score calculation
        if winner_side == "A":
            score = sig_units + finish_units
        elif winner_side == "B":
            score = sig_units - finish_units
        else:
            # Draw or NC
            score = sig_units

        return DominanceResult(
            score=score,
            sig_units=sig_units,
            finish_units=finish_units,
            method_category=method_cat,
            fight_time=fight_time,
            total_time=total_fight_time
        )

    def _compute_finish_units(
            self,
            method_cat: str,
            fight_time: int,
            total_time: int) -> float:
        base = FINISH_BASE_UNITS.get(method_cat, 0.0)

        if base <= 0:
            return 0.0


# bonus for early finishes
        time_affected = {"ko_tko", "sub", "doctor_stoppage", "could_not_continue", "dq"}

        if method_cat in time_affected:
            progress = max(0.0, min(1.0, fight_time / total_time))
            return base * (1.0 + self.weights.EARLY_FINISH_BOOST * (1.0 - progress))

        return base


class EloSystem:
    def __init__(self, config: EloConfig = None):
        self.config = config or EloConfig()
        self.ratings: Dict[str, float] = defaultdict(
            lambda: self.config.INITIAL_RATING
        )
        self.dominance_calc = DominanceCalculator()

    def get_rating(self, fighter: str) -> float:
        return self.ratings[fighter]

    def update(
            self,
            fighter_a: str,
            fighter_b: str,
            winner: Optional[str],
            dominance_score: Optional[float] = None) -> Tuple[float, float]:
        elo_a = self.ratings[fighter_a]
        elo_b = self.ratings[fighter_b]

        #expected scores
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / self.config.SCALE))
        expected_b = 1 - expected_a

        # actual scores
        if winner == fighter_a:
            actual_a, actual_b = 1.0, 0.0
            sign = 1
        elif winner == fighter_b:
            actual_a, actual_b = 0.0, 1.0
            sign = -1
        else:
            actual_a, actual_b = 0.5, 0.5
            sign = 0

        # k factor adjustment
        k = self.config.K
        if dominance_score is not None and sign != 0:
            adj = sign * dominance_score / self.config.SCALE
            adj = max(-1.0, min(1.0, adj))
            k *= (1 + adj)

        # New ratings
        new_elo_a = elo_a + k * (actual_a - expected_a)
        new_elo_b = elo_b + k * (actual_b - expected_b)

        self.ratings[fighter_a] = new_elo_a
        self.ratings[fighter_b] = new_elo_b

        return new_elo_a, new_elo_b

    def get_top_fighters(self, n: int = 20) -> list:
        sorted_fighters = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_fighters[:n]