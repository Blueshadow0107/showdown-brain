"""agent.py — VsAgent: one-ply expectimax, V(s) leaves, pessimistic opponent.

EV(action) = V(state after my action + opponent's best damaging response).
Unmodelable move effects fall back with a small penalty (the 'prior' hook:
swap in pi's argmax later).
"""
import json
import pathlib

import lightgbm as lgb
import numpy as np
from poke_env.player import Player

from sb import features, pi2, transitions as T
from sb.features import state_features, to_id
from sb.showdown_state import battle_to_state, _mon_public

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class VsAgent(Player):
    def __init__(self, *args, log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        features._load()
        T.load()
        self.v = lgb.Booster(model_file=str(ROOT / "models" / "v_model.txt"))
        try:
            self.pi2 = pi2.load()
        except FileNotFoundError:
            self.pi2 = None  # opponent model falls back to pessimistic max-damage
        self._ctx = None     # per-decision cache for the opponent-response model
        self.log_path = log_path
        self._log = open(log_path, "a") if log_path else None

    def _v(self, state: dict) -> float:
        x = np.array([state_features(state)], dtype="float32")
        return float(self.v.predict(x)[0])

    def _opp_ctx(self, state: dict) -> dict:
        """Opponent-response model, cached per decision (choose_move resets).

        pi2 available: distribution over the foe's damaging moves (renormalized)
        weighted by P(move | foe-perspective state). Missing: uniform over the
        same pool (pure max-damage pessimism is gone; the pool itself is the cap).
        """
        if self._ctx is not None:
            return self._ctx
        foe = state["foe"]["active"] or {}
        known = [m for m in foe.get("moves", [])
                 if T.MOVES.get(m, {}).get("category") != "status"]
        pool = known
        if not pool:
            s = set()
            entry = T.SETS.get(foe.get("species", ""))
            if entry:
                for st in entry["sets"]:
                    s.update(to_id(m) for m in st["movepool"])
            pool = [m for m in s
                    if T.MOVES.get(m, {}).get("power")
                    and T.MOVES.get(m, {}).get("category") != "status"][:8] or ["tackle"]
        if self.pi2 is None:
            ctx = {"p_move": 1.0, "weights": {m: 1.0 / len(pool) for m in pool}}
        else:
            flip = {**state, "my": state["foe"], "foe": state["my"]}
            probs = pi2.action_probs(self.pi2, flip, ["move:" + m for m in pool])
            w = {m: probs.get("move:" + m, 1e-6) for m in pool}
            tot = sum(w.values()) or 1.0
            x = np.asarray(state_features(flip), dtype="float32").reshape(1, -1)
            self._ctx = {"p_move": float(self.pi2["kind"].predict(x)[0]),
                         "weights": {m: p / tot for m, p in w.items()}}
            ctx = self._ctx
        self._ctx = ctx
        return ctx

    def _opp_moves(self, state: dict) -> list[str]:
        return list(self._opp_ctx(state)["weights"])

    def _retaliate(self, s2: dict, state: dict):
        me, foe = s2["my"]["active"], s2["foe"]["active"]
        if not me or not foe or foe["hp"] <= 0 or me["hp"] <= 0:
            return
        ctx = self._opp_ctx(state)
        exp = sum(w * T.damage(foe, me, om, s2) for om, w in ctx["weights"].items())
        me["hp"] = max(0.0, me["hp"] - ctx["p_move"] * exp)
        if me["hp"] == 0:
            s2["my"]["fainted"] = min(6, s2["my"]["fainted"] + 1)
            s2["my"]["remaining"] = max(0, s2["my"]["remaining"] - 1)

    def _ev_move(self, state: dict, move_id: str, tera_type: str | None = None) -> float:
        s2 = T.clone(state)
        entry = T.MOVES.get(move_id, {})
        me, foe = s2["my"]["active"], s2["foe"]["active"]
        if tera_type and me:
            me["tera"] = tera_type
        d = 0.0
        if entry.get("category") != "status":
            d = T.damage(me, foe, move_id, s2)
            foe["hp"] = max(0.0, foe["hp"] - d)
            if foe["hp"] == 0:
                s2["foe"]["fainted"] = min(6, s2["foe"]["fainted"] + 1)
                s2["foe"]["remaining"] = max(0, s2["foe"]["remaining"] - 1)
            if entry.get("recoil") and me:
                me["hp"] = max(0.0, me["hp"] - 0.15)
            if entry.get("drain") and me:
                me["hp"] = min(1.0, me["hp"] + d * entry["drain"][0] / entry["drain"][1])
        ko = bool(foe) and foe["hp"] == 0
        modelled = T.apply_move_effects(s2, "my", move_id)
        self._retaliate(s2, state)
        base = self._v(s2)
        if not modelled:
            base -= 0.02
        # endgame urgency: on our last mon, a non-damaging turn we don't
        # survive is a wasted turn — V's coarse features can't price it
        died = not me or me["hp"] == 0
        if state["my"]["remaining"] <= 1 and died and d == 0 and not ko:
            base -= 0.06
        return base

    def _ev_switch(self, state: dict, bench_mon) -> float:
        s2 = T.clone(state)
        incoming = _mon_public(bench_mon)
        T.apply_entry_hazards(incoming, s2["my"].get("hazards", {}))
        if incoming["hp"] == 0:  # hazard death on entry (rocks vs a chipped mon)
            s2["my"]["fainted"] = min(6, s2["my"]["fainted"] + 1)
            s2["my"]["remaining"] = max(0, s2["my"]["remaining"] - 1)
            s2["my"]["active"] = incoming
            return self._v(s2)
        s2["my"]["active"] = incoming
        self._retaliate(s2, state)
        return self._v(s2)

    def choose_move(self, battle):
        state = battle_to_state(battle)
        self._ctx = None  # opponent-response cache lives for one decision
        options = []  # (name, ev, order)
        force = battle.force_switch or not battle.available_moves
        if not force:
            move_opts = []
            for m in battle.available_moves:
                try:
                    ev = self._ev_move(state, m.id)
                except Exception as e:
                    ev = self._v(state) - 0.05
                    move_opts.append((f"move:{m.id}", ev, self.create_order(m)))
                    if self._log:
                        self._log.write(json.dumps({"battle": battle.battle_tag,
                            "turn": battle.turn, "error": f"move:{m.id}: {e!r}"}) + "\n")
                    continue
                move_opts.append((f"move:{m.id}", ev, self.create_order(m)))
            options += move_opts
            # terastallize candidates: re-evaluate top-3 damaging moves with tera
            # typing applied; picked only if they beat the best plain EV by a margin
            ap = battle.active_pokemon
            tera_type = None
            if battle.can_tera and ap is not None:
                if ap.tera_type is not None:
                    tera_type = ap.tera_type.name.title()
                else:
                    tera_type = T.tera_prior(to_id(ap.species))
            if tera_type and move_opts:
                plain_best = max(ev for _, ev, _ in move_opts)
                dmg = [o for o in move_opts
                       if T.MOVES.get(o[0][5:], {}).get("category") != "status"]
                for name, _, _ in sorted(dmg, key=lambda x: x[1], reverse=True)[:3]:
                    move_id = name[5:]
                    try:
                        ev = self._ev_move(state, move_id, tera_type=tera_type)
                    except Exception:
                        continue
                    if ev > plain_best + 0.03:
                        m = next(m for m in battle.available_moves if m.id == move_id)
                        options.append((f"move:{move_id}|tera:{tera_type}", ev,
                                        self.create_order(m, terastallize=True)))
        for s in battle.available_switches:
            try:
                ev = self._ev_switch(state, s)
            except Exception as e:
                ev = self._v(state) - 0.05
                options.append((f"switch:{to_id(s.species)}", ev, self.create_order(s)))
                if self._log:
                    self._log.write(json.dumps({"battle": battle.battle_tag,
                        "turn": battle.turn, "error": f"switch:{s.species}: {e!r}"}) + "\n")
                continue
            options.append((f"switch:{to_id(s.species)}", ev, self.create_order(s)))
        if not options:
            return self.choose_random_move(battle)
        options.sort(key=lambda x: x[1], reverse=True)
        if self._log:
            self._log.write(json.dumps({
                "battle": battle.battle_tag, "turn": battle.turn,
                "chosen": options[0][0],
                "options": {n: round(e, 4) for n, e, _ in options},
            }) + "\n")
            self._log.flush()
        return options[0][2]
