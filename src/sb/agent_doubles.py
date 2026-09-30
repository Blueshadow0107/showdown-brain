"""agent_doubles.py — VsAgentDoubles: one-ply expectimax for gen 9 random doubles.

Mirror of VsAgent for DoubleBattle: per active slot, EV(action) = V(state after
my action + both foe actives' expected damaging response), leaves scored by the
doubles V model (features_doubles). Differences from singles, by design (v1):

- Opponent model is uniform over each foe active's damaging move pool (from
  randdoubles-sets.json); no pi2 weighting (pi2 is singles-dim).
- Spread moves (target allAdjacent/allAdjacentFoes) deal 0.75x damage to EACH
  foe active; allAdjacent's ally damage is NOT modelled (simplification).
- Retaliation sums expected damage from BOTH foe actives onto the acting slot.
- No terastallize candidates yet.
"""
import copy
import json
import pathlib

import lightgbm as lgb
import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder

from sb import features, features_doubles as FD, transitions as T
from sb.features import to_id
from sb.showdown_state_doubles import battle_to_state_d, _mon_public_d

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

MAX_CANDIDATES_PER_SLOT = 8  # decision budget: sorted by rough power, then switches

SETS_D = None


def _doubles_sets() -> dict:
    global SETS_D
    if SETS_D is None:
        SETS_D = json.load(open(features.DATA / "randdoubles-sets.json"))
    return SETS_D


class VsAgentDoubles(Player):
    def __init__(self, *args, log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        features._load()
        T.load()
        self.v = lgb.Booster(model_file=str(ROOT / "models" / "v_model_doubles.txt"))
        self._pools = {}     # per-decision cache: foe pos -> damaging move pool
        self.log_path = log_path
        self._log = open(log_path, "a") if log_path else None

    def _v(self, state: dict, slot: str) -> float:
        x = np.array([FD.state_features_d(state, slot)], dtype="float32")
        return float(self.v.predict(x)[0])

    def _foe_pool(self, foe: dict) -> list[str]:
        """Damaging-move pool for one foe active mon (uniform weights, v1).

        Revealed non-status moves if any, else the doubles sets-file movepool.
        Cached per decision (choose_move resets).
        """
        key = (foe.get("pos"), tuple(foe.get("moves", [])))
        if key in self._pools:
            return self._pools[key]
        pool = [m for m in foe.get("moves", [])
                if T.MOVES.get(m, {}).get("category") != "status"]
        if not pool:
            s = set()
            entry = _doubles_sets().get(foe.get("species", ""))
            if entry:
                for st in entry["sets"]:
                    s.update(to_id(m) for m in st["movepool"])
            pool = [m for m in s
                    if T.MOVES.get(m, {}).get("power")
                    and T.MOVES.get(m, {}).get("category") != "status"][:8] or ["tackle"]
        self._pools[key] = pool
        return pool

    def _retaliate_d(self, s2: dict, slot_i: int):
        """Expected damage on my slot-i mon from EACH living foe active."""
        me = s2["my"]["active"][slot_i]
        if not me or me["hp"] <= 0:
            return
        exp_total = 0.0
        for foe in s2["foe"]["active"]:
            if not foe or foe["hp"] <= 0:
                continue
            pool = self._foe_pool(foe)
            exp_total += sum(T.damage(foe, me, om, s2) for om in pool) / len(pool)
        me["hp"] = max(0.0, me["hp"] - exp_total)
        if me["hp"] == 0:
            s2["my"]["fainted"] = min(6, s2["my"]["fainted"] + 1)
            s2["my"]["remaining"] = max(0, s2["my"]["remaining"] - 1)

    @staticmethod
    def _faint_foe(s2: dict, foe: dict):
        if foe["hp"] == 0:
            s2["foe"]["fainted"] = min(6, s2["foe"]["fainted"] + 1)
            s2["foe"]["remaining"] = max(0, s2["foe"]["remaining"] - 1)

    def _apply_effects_d(self, s2: dict, slot_i: int, move_id: str,
                         target: dict | None) -> bool:
        """Doubles-adapted copy of transitions.apply_move_effects (that one
        assumes side["active"] is a single mon dict)."""
        entry = T.MOVES.get(move_id)
        if not entry:
            return False
        me = s2["my"]["active"][slot_i]
        handled = False
        if entry.get("boosts") and me:
            T.apply_boosts(me, entry["boosts"])
            handled = True
        self_eff = entry.get("self") or {}
        if isinstance(self_eff, dict) and self_eff.get("boosts") and me:
            T.apply_boosts(me, self_eff["boosts"])
            handled = True
        if entry.get("heal") and me:
            me["hp"] = min(1.0, me["hp"] + entry["heal"][0] / entry["heal"][1])
            handled = True
        if entry.get("status") and target:
            if not target.get("status"):
                target["status"] = entry["status"]
            handled = True
        if entry.get("sideCondition"):
            side = s2["foe"]
            cond = entry["sideCondition"]
            side["hazards"][cond] = side["hazards"].get(cond, 0) + 1
            handled = True
        if entry.get("weather"):
            s2["weather"] = {"raindance": "raindance", "sunnyday": "sunnyday",
                             "sandstorm": "sandstorm", "snowscape": "snow"}.get(entry["weather"])
            handled = True
        if entry.get("terrain"):
            s2["terrain"] = entry["terrain"]
            handled = True
        return handled

    def _ev_move_d(self, state: dict, slot_i: int, move_id: str,
                   foes_alive: list[int]) -> tuple[float, int | None]:
        """EV of slot-i mon using move_id. Returns (ev, target_idx) where
        target_idx is the foe active slot to aim at (None = no target)."""
        s2 = T.clone(state)
        me = s2["my"]["active"][slot_i]
        foes = s2["foe"]["active"]
        entry = T.MOVES.get(move_id, {})
        target_idx = foes_alive[0] if foes_alive else None
        target = foes[target_idx] if target_idx is not None else None
        d = 0.0
        if entry.get("category") != "status" and entry.get("power"):
            if entry.get("target") in ("allAdjacent", "allAdjacentFoes"):
                # spread: 0.75x to each foe active; ally damage not modelled (v1)
                for fi in foes_alive:
                    foe = foes[fi]
                    d = T.damage(me, foe, move_id, s2) * 0.75
                    foe["hp"] = max(0.0, foe["hp"] - d)
                    self._faint_foe(s2, foe)
                target_idx, target = None, None
            elif target is not None:
                # single target: aim at the foe active taking max damage
                dmg = {fi: T.damage(me, foes[fi], move_id, s2) for fi in foes_alive}
                target_idx = max(dmg, key=dmg.get)
                target = foes[target_idx]
                d = dmg[target_idx]
                target["hp"] = max(0.0, target["hp"] - d)
                self._faint_foe(s2, target)
            if entry.get("recoil") and me:
                me["hp"] = max(0.0, me["hp"] - 0.15)
            if entry.get("drain") and me:
                me["hp"] = min(1.0, me["hp"] + d * entry["drain"][0] / entry["drain"][1])
        modelled = self._apply_effects_d(s2, slot_i, move_id, target)
        self._retaliate_d(s2, slot_i)
        base = self._v(s2, me["pos"] if me else None)
        return (base if modelled else base - 0.02), target_idx

    def _ev_switch_d(self, state: dict, slot_i: int, bench_mon: dict,
                     slot_pos: str) -> float:
        s2 = T.clone(state)
        s2["my"]["active"][slot_i] = copy.deepcopy(bench_mon)
        self._retaliate_d(s2, slot_i)
        return self._v(s2, slot_pos)

    def choose_move(self, battle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        state = battle_to_state_d(battle)
        self._pools = {}
        slot_orders = []
        slot_log = []
        slot_opts = []
        for i in range(2):
            letter = "ab"[i]
            pos = f"{battle.player_role}{letter}"
            force = battle.force_switch[i]
            if any(battle.force_switch) and not force:
                # replacement phase: the non-forced slot must pass
                slot_orders.append(PassBattleOrder())
                slot_log.append({"chosen": "pass", "forced_pass": True})
                continue
            moves = list(battle.available_moves[i])
            switches = list(battle.available_switches[i])
            foes_alive = [fi for fi, f in enumerate(battle.opponent_active_pokemon)
                          if f is not None and not f.fainted]
            cands = []  # (rough_power, name, kind, payload)
            if not force:
                for m in moves:
                    cands.append((T.MOVES.get(m.id, {}).get("power") or 0,
                                  f"move:{m.id}", "move", m))
            for s in switches:
                cands.append((-1, f"switch:{to_id(s.species)}", "switch", s))
            cands.sort(key=lambda c: c[0], reverse=True)
            cands = cands[:MAX_CANDIDATES_PER_SLOT]
            if not cands:
                slot_orders.append(PassBattleOrder())
                slot_log.append({"chosen": "pass", "no_candidates": True})
                continue
            opts, err = [], None
            for _, name, kind, payload in cands:
                try:
                    if kind == "move":
                        ev, t_idx = self._ev_move_d(state, i, payload.id, foes_alive)
                        order = self.create_order(
                            payload,
                            move_target=(battle.to_showdown_target(payload, None)
                                         if t_idx is None else
                                         battle.to_showdown_target(
                                             payload,
                                             battle.opponent_active_pokemon[t_idx])))
                    else:
                        pub = _mon_public_d(payload, pos)
                        ev = self._ev_switch_d(state, i, pub, pos)
                        order = self.create_order(payload)
                    opts.append((name, ev, order))
                except Exception as e:
                    err = f"{name}: {e!r}"
                    opts.append((name, self._v(state, pos) - 0.05,
                                 self.create_order(payload)))
            opts.sort(key=lambda x: x[1], reverse=True)
            slot_opts.append(opts)
            slot_orders.append(opts[0][2])
            entry = {"chosen": opts[0][0],
                     "options": {n: round(e, 4) for n, e, _ in opts}}
            if err:
                entry["error"] = err
            slot_log.append(entry)
        # double replacement: both slots may pick the same bench mon —
        # join_orders would drop the pair, so demote slot1 to its runner-up
        if (len(slot_opts) == 2 and slot_opts[0] and slot_opts[1]
                and getattr(slot_orders[0].order, "name", None)
                and slot_orders[0].order is slot_orders[1].order):
            if len(slot_opts[1]) > 1:
                slot_orders[1] = slot_opts[1][1][2]
                slot_log[1]["chosen"] = slot_opts[1][1][0]
                slot_log[1]["demoted_duplicate"] = True
            elif len(slot_opts[0]) > 1:
                slot_orders[0] = slot_opts[0][1][2]
                slot_log[0]["chosen"] = slot_opts[0][1][0]
                slot_log[0]["demoted_duplicate"] = True
        if self._log:
            self._log.write(json.dumps({
                "battle": battle.battle_tag, "turn": battle.turn,
                "slot0": slot_log[0], "slot1": slot_log[1],
            }) + "\n")
            self._log.flush()
        orders = DoubleBattleOrder.join_orders([slot_orders[0]], [slot_orders[1]])
        if not orders:
            return Player.choose_random_doubles_move(battle)
        return orders[0]
