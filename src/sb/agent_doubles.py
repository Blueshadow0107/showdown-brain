"""agent_doubles.py — VsAgentDoubles: one-ply expectimax for gen 9 random doubles.

Mirror of VsAgent for DoubleBattle: per active slot, EV(action) = V(state after
my action + both foe actives' expected damaging response), leaves scored by the
doubles V model (features_doubles). Differences from singles, by design (v1):

- Opponent model: pi2d (doubles policy prior) weights over each foe slot's
  damaging move pool, scaled by P(move | foe-perspective state) — so protect-
  heavy or switch-happy foe states naturally discount the expected hit.
- Spread moves (target allAdjacent/allAdjacentFoes) deal 0.75x damage to EACH
  foe active; allAdjacent's ally damage is NOT modelled (simplification).
- Retaliation sums expected damage from BOTH foe actives onto the acting slot.
- No terastallize candidates yet.
"""
import copy
import json
import pathlib
import traceback

import lightgbm as lgb
import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder

from sb import features, features_doubles as FD, pi2doubles, transitions as T
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
        try:
            self.pi2d = pi2doubles.load()
        except FileNotFoundError:
            self.pi2d = None  # retaliation falls back to uniform pool weights
        self._pools = {}        # per-decision cache: foe pos -> damaging move pool
        self._foe_pi_cache = {}  # per-decision cache: foe pos -> (weights, p_move)
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

    def _foe_pi(self, foe: dict, state: dict, pool: list):
        """pi2d-weighted response for one foe slot, cached per decision.

        Returns (weights {move: p}, p_move). p_move < 1 means the prior expects
        a switch (or another non-move order) from that slot often enough to
        matter. Falls back to uniform when the bundle is missing.
        """
        key = foe.get("pos")
        if key in self._foe_pi_cache:
            return self._foe_pi_cache[key]
        if self.pi2d is None:
            ctx = ({m: 1.0 / len(pool) for m in pool}, 1.0)
        else:
            flip = {**state, "my": state["foe"], "foe": state["my"]}
            slot = foe.get("pos")
            probs = pi2doubles.action_probs(self.pi2d, flip, slot,
                                            ["move:" + m for m in pool])
            w = {m: probs.get("move:" + m, 1e-6) for m in pool}
            tot = sum(w.values()) or 1.0
            x = np.array([FD.state_features_d(flip, slot)], dtype="float32")
            p_move = float(self.pi2d["kind"].predict(x)[0])
            ctx = ({m: p / tot for m, p in w.items()}, p_move)
        self._foe_pi_cache[key] = ctx
        return ctx

    def _retaliate_d(self, s2: dict, slot_i: int, state: dict):
        """Expected damage on my slot-i mon from each living foe active,
        weighted by the pi2d response prior (protect-heavy states discount)."""
        me = s2["my"]["active"][slot_i]
        if not me or me["hp"] <= 0:
            return
        exp_total = 0.0
        for foe in s2["foe"]["active"]:
            if not foe or foe["hp"] <= 0:
                continue
            pool = self._foe_pool(foe)
            weights, p_move = self._foe_pi(foe, state, pool)
            exp_total += p_move * sum(w * T.damage(foe, me, om, s2)
                                      for om, w in weights.items())
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
            status = entry["status"]
            type_immune = T.effectiveness(entry["type"], target) == 0
            status_immune = (status == "par" and "Electric" in T.defender_types(target)) \
                or (status == "brn" and ("Fire" in T.defender_types(target)))
            if target.get("status") or type_immune or status_immune:
                return False  # wasted turn: no effect lands
            target["status"] = status
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
                   foes_alive: list[int]) -> tuple:
        """EV of slot-i mon using move_id. Returns (ev, target_idx, dmg, ko, died)
        where dmg is total HP dealt, ko = a foe dropped, died = the acting mon
        faints to the expected retaliation."""
        s2 = T.clone(state)
        me = s2["my"]["active"][slot_i]
        foes = s2["foe"]["active"]
        entry = T.MOVES.get(move_id, {})
        target_idx = foes_alive[0] if foes_alive else None
        target = foes[target_idx] if target_idx is not None else None
        d = 0.0
        if entry.get("category") != "status" and entry.get("power"):
            if entry.get("target") in ("allAdjacent", "allAdjacentFoes"):
                # spread: 0.75x to each foe active; allAdjacent also clips the ally
                for fi in foes_alive:
                    foe = foes[fi]
                    d_spread = T.damage(me, foe, move_id, s2) * 0.75
                    d += d_spread
                    foe["hp"] = max(0.0, foe["hp"] - d_spread)
                    self._faint_foe(s2, foe)
                if entry.get("target") == "allAdjacent":
                    ally = s2["my"]["active"][1 - slot_i]
                    if ally and ally["hp"] > 0:
                        d_ally = T.damage(me, ally, move_id, s2) * 0.75
                        ally["hp"] = max(0.0, ally["hp"] - d_ally)
                        if ally["hp"] == 0:
                            s2["my"]["fainted"] = min(6, s2["my"]["fainted"] + 1)
                            s2["my"]["remaining"] = max(0, s2["my"]["remaining"] - 1)
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
        ko = any(f and f["hp"] == 0 for f in foes)
        modelled = self._apply_effects_d(s2, slot_i, move_id, target)
        self._retaliate_d(s2, slot_i, state)
        died = not me or me["hp"] == 0
        base = self._v(s2, me["pos"] if me else None)
        ev = base if modelled else base - 0.02
        # endgame urgency: on our last mon, a setup turn we don't survive is a
        # wasted turn — V's coarse features can't see it, so correct directly
        if state["my"]["remaining"] <= 1 and died and d == 0 and not ko:
            ev -= 0.06
        return ev, target_idx, d, ko, died

    def _ev_switch_d(self, state: dict, slot_i: int, bench_mon: dict,
                     slot_pos: str) -> float:
        s2 = T.clone(state)
        s2["my"]["active"][slot_i] = copy.deepcopy(bench_mon)
        self._retaliate_d(s2, slot_i, state)
        return self._v(s2, slot_pos)

    def choose_move(self, battle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        # momentum counters: reset per battle; mine counted on switch orders,
        # foe's by changes to the pair of active species between decisions
        if getattr(self, "_battle_tag", None) != battle.battle_tag:
            self._battle_tag = battle.battle_tag
            self._sw = {"my": 0, "foe": 0}
            self._last_foe_pair = None
        foe_pair = tuple(sorted(to_id(m.species) for m in battle.opponent_active_pokemon
                                if m is not None and not m.fainted))
        if self._last_foe_pair is not None and foe_pair != self._last_foe_pair:
            changed = len(set(foe_pair) - set(self._last_foe_pair))
            self._sw["foe"] += max(1, changed)
        self._last_foe_pair = foe_pair
        state = battle_to_state_d(battle, switch_counts=self._sw)
        self._pools = {}
        self._foe_pi_cache = {}
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
                # empty slot with no legal orders: passable only during forced
                # replacement; otherwise random fallback keeps the joint order legal
                if force:
                    slot_orders.append(PassBattleOrder())
                    slot_log.append({"chosen": "pass", "forced_pass": True})
                else:
                    return Player.choose_random_doubles_move(battle)
                continue
            opts, err, dmg_by_name = [], None, {}
            for _, name, kind, payload in cands:
                try:
                    if kind == "move":
                        ev, t_idx, d, _, _ = self._ev_move_d(state, i, payload.id, foes_alive)
                        dmg_by_name[name] = d
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
                except Exception:
                    err = traceback.format_exc(limit=4)
                    continue  # skip poisoned candidate — never let one brick the slot
            if not opts:
                slot_orders.append(PassBattleOrder())
                slot_log.append({"chosen": "pass", "all_candidates_error": err})
                continue
            # near-ties break toward immediate damage — V's coarse features
            # can't value tempo, so when EVs agree, act
            opts.sort(key=lambda x: (x[1], dmg_by_name.get(x[0], 0.0)), reverse=True)
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
        self._sw["my"] += sum(1 for e in slot_log if e["chosen"].startswith("switch:"))
        if self._log:
            self._log.write(json.dumps({
                "battle": battle.battle_tag, "turn": battle.turn,
                "slot0": slot_log[0], "slot1": slot_log[1],
            }) + "\n")
            self._log.flush()
        try:
            orders = DoubleBattleOrder.join_orders([slot_orders[0]], [slot_orders[1]])
            if not orders:
                return Player.choose_random_doubles_move(battle)
            return orders[0]
        except Exception:
            if self._log:
                self._log.write(json.dumps({
                    "battle": battle.battle_tag, "turn": battle.turn,
                    "join_error": traceback.format_exc(limit=4)}) + "\n")
            return Player.choose_random_doubles_move(battle)
