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
    # sunk-boost debit: each positive stage on the outgoing mon taxes a
    # switch — the setup turns it spent die with the boost reset
    sunk_boost_debit: float = 0.015
    # setup-no-gain tax: a self-boost that flips no KO clock (vs multiscale
    # walls) costs tax * (1 + stages invested) — repeated setup escalates
    setup_no_gain_tax: float = 0.03
    # candidate cap per slot in doubles (decision budget)
    doubles_max_candidates: int = 8
    # --- FFA (pairwise projection; VsAgentFFA) ---
    # EV(a) = sum_i w_i * V(next vs foe_i) - ffa_retaliation_weight * sum_j
    # retaliation_j + bonuses. w_i = 1 + ffa_threat_weight * (threat_i -
    # mean(threat)) piles pressure onto the player with the strongest
    # remaining board (gang up on the leader)
    ffa_threat_weight: float = 0.5
    # flat EV bonus when an action KOs an opponent's active (vulture KOs)
    ffa_ko_bonus: float = 0.05
    # bonus * (1 - hp fraction) for damaging an already-hurt target (focus
    # fire beats spreading chip damage)
    ffa_focus_bonus: float = 0.03
    # scales the summed per-opponent expected-retaliation damage term;
    # 3 foes' worth of incoming damage must not dominate a 1v1-calibrated V
    ffa_retaliation_weight: float = 0.3
    # fraction of faster foes' expected damage likely to land on me (each
    # foe picks among three targets) — scales the fizzle check; without it
    # every low-HP turn "fizzles" and Protect/switch spam wins
    ffa_gangup_frac: float = 0.34


H = Heuristics()
