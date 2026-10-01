"""heuristics.py — the one place where gameplay tuning constants live.

The expectimax machinery (features/transitions/models) is learned or physical;
these are the deliberate judgment calls on top of it. Change them here, not
in the agents — the agents import from this module only.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Heuristics:
    # switch-commit: a voluntary pivot must beat the best move by this margin
    # (1-ply sims underprice the entry hit + tempo + double-switch risk)
    switch_margin: float = 0.025
    # repetition tax: each consecutive repeat of the same non-damaging move
    # decays its EV by base * (1 + streak) — breaks recovery-stall loops
    repeat_tax_base: float = 0.04
    # endgame urgency: on our last mon, a non-damaging non-KO turn we don't
    # survive is penalized (setup-and-die, e.g. tailwind on a dying last mon)
    urgency_penalty: float = 0.06
    # terastallize candidates must beat the best plain EV by this margin
    tera_margin: float = 0.03
    # unmodelable move effects fall back with this penalty
    unmodelled_penalty: float = 0.02
    # candidate cap per slot in doubles (decision budget)
    doubles_max_candidates: int = 8


H = Heuristics()
