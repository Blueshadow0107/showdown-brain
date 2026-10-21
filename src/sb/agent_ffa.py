"""agent_ffa.py — VsAgentFFA: one-ply expectimax for gen 9 free-for-all.

Mirror of VsAgent for FFABattle: the 4-player game is projected into one
ordinary singles-shaped state per live opponent (showdown_state_ffa), each
scored by the v2 V model, action EVs aggregated with the FFA heuristics:

    EV(a) = sum_i w_i * V(next state vs foe_i)
            - H.ffa_retaliation_weight * sum_j retaliation_j
            + H.ffa_ko_bonus per foe active KO'd
            + H.ffa_focus_bonus * (1 - hp) on already-hurt targets

w_i = 1 + H.ffa_threat_weight * (threat_i - mean(threat)), threat_i = foe_i's
remaining board strength (unfainted count + active hp fraction).
retaliation_j is VsAgent._retaliate computed per live foe j: pi2-weighted
expected damage from foe j's active onto mine. Foes that outspeed me strike
before my move — if their combined expected damage KOs my active the action
fizzles (VsAgent._ev_move's foe_first branch). Targeted moves enumerate one
action per live opponent (target loc from sb.ffa.target_loc); spread moves
clip every live foe at sb.ffa.SPREAD_MODIFIER; other untargeted damaging
moves are modelled against the max-damage live foe. Move effects apply
pairwise per projection like VsAgentDoubles._apply_effects_d, never the
2-side flip in transitions.apply_move_effects. Brains follow the VsAgent
convention {"model", "featurizer", "pi2"}; the default model is
models/v_ffa.txt when present, else the v2 build.
"""
import json
import pathlib

import lightgbm as lgb
import numpy as np
from poke_env.player import Player

from sb import features, features_v2, pi2, transitions as T
from sb.features import state_features, to_id
from sb.ffa import SPREAD_MODIFIER, needs_target, target_loc
from sb.ffa_poke_env import FFABattle
from sb.heuristics import H
from sb.showdown_state import _mon_public
from sb.showdown_state_ffa import battle_to_state_ffa

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class VsAgentFFA(Player):
    def __init__(self, *args, log_path=None, brain: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        features._load()
        T.load()
        brain = brain or {}
        featurizer = brain.get("featurizer") or "v2"
        if featurizer == "v2":
            self._featurize = features_v2.state_features_v2
        elif featurizer == "v1-classic":
            # pre-momentum models (287 dims): v1 vector minus the 13-dim
            # momentum block appended at the end of state_features
            self._featurize = lambda s: state_features(s)[:-features.MOMENTUM_DIM]
        else:
            self._featurize = state_features
        model = brain.get("model")
        if model is None:
            model = ("v_ffa.txt" if (ROOT / "models" / "v_ffa.txt").exists()
                     else "v2_model.txt")
        self.v = lgb.Booster(model_file=str(ROOT / "models" / model))
        if brain.get("pi2", True):
            try:
                self.pi2 = pi2.load()
            except FileNotFoundError:
                self.pi2 = None  # opponent model falls back to uniform pool weights
        else:
            self.pi2 = None
        self._ctxs = {}      # per-decision cache: foe_n -> opponent-response ctx
        self._rets = {}      # per-decision cache: foe_n -> expected retaliation dmg
        self._last_move = {} # battle tag -> my previous move id (repeat tax)
        self.log_path = log_path
        self._log = open(log_path, "a") if log_path else None

    def _ensure_attrs(self):
        """Same daemon-safety guard as VsAgent: poke-env can invoke choose_move
        on an instance whose __init__ was interrupted; never let a missing
        attr brick a live battle."""
        for name, default in (("pi2", None), ("_log", None), ("_ctxs", None),
                              ("_rets", None), ("_sw", None),
                              ("_battle_tag", None), ("_last_foes", None)):
            if not hasattr(self, name):
                setattr(self, name, default)

    def _v(self, state: dict) -> float:
        x = np.array([self._featurize(state)], dtype="float32")
        return float(self.v.predict(x)[0])

    # ---------- teampreview ----------

    def teampreview(self, battle) -> str:
        """Lead with the highest total-base-stats mon, rest in stats order."""
        team = list(battle.team.values())
        if not team:
            return "/team 123456"

        def bst(mon) -> int:
            sp = T.SPECIES.get(features.resolve_species(to_id(mon.species)), {})
            bs = sp.get("baseStats") or {}
            return sum(bs.get(s, 80) for s in
                       ("hp", "atk", "def", "spa", "spd", "spe"))

        ordered = sorted(team, key=bst, reverse=True)
        for mon in ordered:
            mon._selected_in_teampreview = True
        return "/team " + "".join(str(i + 1) for i in
                                  (team.index(m) for m in ordered))

    # ---------- opponent response ----------

    def _opp_ctx(self, state: dict, foe_n: int) -> dict:
        """Opponent-response model for one foe, cached per decision.

        Same construction as VsAgent._opp_ctx: pi2-weighted damaging-move
        pool (revealed moves, else the active randbats sets table — 'ffa'
        while a choose_move is in flight), uniform fallback without pi2.
        """
        if foe_n in self._ctxs:
            return self._ctxs[foe_n]
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
            ctx = {"p_move": float(self.pi2["kind"].predict(x)[0]),
                   "weights": {m: p / tot for m, p in w.items()}}
        self._ctxs[foe_n] = ctx
        return ctx

    def _retaliation(self, projs: dict, view) -> float:
        """Expected hp-fraction damage view's active deals onto my active,
        pi2-weighted (VsAgent._retaliate for a single foe, state unmutated).
        Cached per decision."""
        if view.n in self._rets:
            return self._rets[view.n]
        state = projs[view.n]
        me, foe = state["my"]["active"], state["foe"]["active"]
        exp = 0.0
        if me and foe and me["hp"] > 0 and foe["hp"] > 0:
            ctx = self._opp_ctx(state, view.n)
            exp = ctx["p_move"] * sum(
                w * T.damage(foe, me, om, state,
                             defender_screens=state["my"].get("screens"))
                for om, w in ctx["weights"].items())
        self._rets[view.n] = exp
        return exp

    def _retaliate_into(self, s2: dict, state: dict, foe_n: int):
        """VsAgent._retaliate onto s2's my active from foe foe_n (mutates s2) —
        used after switches, where the incoming mon absorbs every foe's hit."""
        me = s2["my"]["active"]
        if not me or me["hp"] <= 0:
            return
        foe = state["foe"]["active"]
        if not foe or foe["hp"] <= 0:
            return
        ctx = self._opp_ctx(state, foe_n)
        exp = sum(w * T.damage(foe, me, om, state,
                               defender_screens=s2["my"].get("screens"))
                  for om, w in ctx["weights"].items())
        me["hp"] = max(0.0, me["hp"] - ctx["p_move"] * exp)
        if me["hp"] == 0:
            s2["my"]["fainted"] = min(6, s2["my"]["fainted"] + 1)
            s2["my"]["remaining"] = max(0, s2["my"]["remaining"] - 1)

    @staticmethod
    def _threat(view) -> float:
        """Remaining board strength: unfainted count plus the active's hp
        fraction (0 when the side has no active out)."""
        active = view.active
        hp = float(active.current_hp_fraction) \
            if active is not None and not active.fainted else 0.0
        return view.pokemon_left + hp

    def _weights(self, foes) -> dict:
        """EV aggregation weights — gang up on the leader: w_i rises with
        foe_i's board strength above the live-foe mean."""
        threats = {v.n: self._threat(v) for v in foes}
        mean = sum(threats.values()) / len(foes)
        return {n: 1.0 + H.ffa_threat_weight * (t - mean)
                for n, t in threats.items()}

    # ---------- move effects ----------

    @staticmethod
    def _apply_effects_ffa(s2: dict, move_id: str, target: dict | None) -> bool:
        """Pairwise copy of transitions.apply_move_effects for one foe
        projection — that one flips the two sides internally and assumes the
        move's target is the state's foe active (pattern from
        VsAgentDoubles._apply_effects_d)."""
        entry = T.MOVES.get(move_id)
        if not entry:
            return False
        me = s2["my"]["active"]
        handled = False
        if entry.get("boosts") and me:  # self-boosting status (Swords Dance etc.)
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
            # a status that can't land (target already statused, type-immune
            # to the move, or para-vs-electric) is a WASTED TURN
            status = entry["status"]
            type_immune = T.effectiveness(entry["type"], target) == 0
            status_immune = (status == "par" and "Electric" in T.defender_types(target)) \
                or (status == "brn" and ("Fire" in T.defender_types(target)))
            if target.get("status") or type_immune or status_immune:
                return False
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

    # ---------- action EVs ----------

    def _ev_move_ffa(self, projs: dict, foes, weights: dict, rets: dict,
                     move_id: str, target_n: int | None,
                     tera_type: str | None = None,
                     repeat: bool = False) -> float:
        state0 = projs[target_n] if target_n is not None \
            else next(iter(projs.values()))
        me0 = state0["my"]["active"]
        entry = T.MOVES.get(move_id, {})
        prio = entry.get("priority") or 0
        # foes acting before my move (no priority, faster under current speed)
        first = [v.n for v in foes
                 if me0 and prio <= 0 and projs[v.n]["foe"]["active"]
                 and projs[v.n]["foe"]["active"]["hp"] > 0
                 and T.effective_speed(projs[v.n]["foe"]["active"], projs[v.n])
                 > T.effective_speed(me0, state0)]
        if me0 and me0["hp"] > 0 \
                and sum(rets[n] for n in first) * H.ffa_gangup_frac >= me0["hp"]:
            # fizzled: enough focused faster-foe damage drops my active
            # before acting, the move never lands
            tot = 0.0
            for v in foes:
                s2 = T.clone(projs[v.n])
                m2 = s2["my"]["active"]
                if m2 and m2["hp"] > 0:
                    m2["hp"] = 0.0
                    s2["my"]["fainted"] = min(6, s2["my"]["fainted"] + 1)
                    s2["my"]["remaining"] = max(0, s2["my"]["remaining"] - 1)
                ev_i = self._v(s2)
                if state0["my"]["remaining"] <= 1:
                    ev_i -= H.urgency_penalty
                tot += weights[v.n] * ev_i
            return tot
        spread = entry.get("target") in ("allAdjacent", "allAdjacentFoes")
        hit_n = target_n
        if hit_n is None and entry.get("category") != "status" \
                and entry.get("power") and not spread:
            # untargeted damaging move: the server auto-picks, model the
            # max-damage live foe (VsAgentDoubles' single-target judgment)
            dmg = {v.n: T.damage(state0["my"]["active"],
                                 projs[v.n]["foe"]["active"], move_id,
                                 projs[v.n],
                                 defender_screens=projs[v.n]["foe"].get("screens"))
                   for v in foes if projs[v.n]["foe"]["active"]}
            hit_n = max(dmg, key=dmg.get) if dmg and max(dmg.values()) > 0 else None
        tot, bonus, total_d = 0.0, 0.0, 0.0
        ko_any = modelled_any = recoil_died = False
        for v in foes:
            s2 = T.clone(projs[v.n])
            m2, f2 = s2["my"]["active"], s2["foe"]["active"]
            if tera_type and m2:
                m2["tera"] = tera_type
            hits = spread or (hit_n is not None and v.n == hit_n)
            d = 0.0
            if hits and f2 and entry.get("category") != "status" and entry.get("power"):
                d = T.damage(m2, f2, move_id, s2,
                             defender_screens=s2["foe"].get("screens"))
                if spread:
                    d *= SPREAD_MODIFIER
                if f2["hp"] < 1.0:  # focus fire: reward hurting the wounded
                    bonus += H.ffa_focus_bonus * (1.0 - f2["hp"])
                f2["hp"] = max(0.0, f2["hp"] - d)
                total_d += d
                if f2["hp"] == 0:
                    ko_any = True
                    bonus += H.ffa_ko_bonus
                    s2["foe"]["fainted"] = min(6, s2["foe"]["fainted"] + 1)
                    s2["foe"]["remaining"] = max(0, s2["foe"]["remaining"] - 1)
                if entry.get("recoil") and m2:
                    m2["hp"] = max(0.0, m2["hp"] - 0.15)
                    recoil_died = recoil_died or m2["hp"] == 0
                if entry.get("drain") and m2:
                    m2["hp"] = min(1.0, m2["hp"] + d * entry["drain"][0] / entry["drain"][1])
            modelled_any = self._apply_effects_ffa(s2, move_id, f2 if hits else None) \
                or modelled_any
            tot += weights[v.n] * self._v(s2)
        tot += bonus
        tot -= H.ffa_retaliation_weight * sum(rets.values())
        if not modelled_any:
            tot -= H.unmodelled_penalty
        if repeat and not entry.get("power"):
            # Protect fails if repeated, Wish/Sub decay — VsAgent's repeat tax
            tot -= H.repeat_tax_base
        # endgame urgency: on our last mon, a non-damaging turn we don't
        # survive is a wasted turn (recoil death is the only in-state death
        # on this path — first-strike KOs take the fizzle branch)
        if state0["my"]["remaining"] <= 1 and recoil_died \
                and total_d == 0 and not ko_any:
            tot -= H.urgency_penalty
        return tot

    def _ev_switch_ffa(self, projs: dict, foes, weights: dict,
                       bench_mon) -> float:
        stages = sum(v for v in ((next(iter(projs.values()))["my"]["active"]) or {})
                     .get("boosts", {}).values() if v > 0)
        tot = 0.0
        for v in foes:
            s2 = T.clone(projs[v.n])
            incoming = _mon_public(bench_mon)
            T.apply_entry_hazards(incoming, s2["my"].get("hazards", {}))
            if incoming["hp"] == 0:  # hazard death on entry (rocks vs a chip)
                s2["my"]["fainted"] = min(6, s2["my"]["fainted"] + 1)
                s2["my"]["remaining"] = max(0, s2["my"]["remaining"] - 1)
                s2["my"]["active"] = incoming
            else:
                s2["my"]["active"] = incoming
                for w in foes:  # the pivot absorbs every live foe's hit
                    self._retaliate_into(s2, projs[w.n], w.n)
            tot += weights[v.n] * (self._v(s2) - H.sunk_boost_debit * stages)
        return tot

    # ---------- decision ----------

    def choose_move(self, battle):
        if not isinstance(battle, FFABattle) or battle.my_side_n is None:
            return self.choose_random_move(battle)
        self._ensure_attrs()
        T.set_kind("ffa")
        # momentum counters: reset per battle; my switches counted on choice,
        # each foe's inferred from its active-species changes between decisions
        if getattr(self, "_battle_tag", None) != battle.battle_tag:
            self._battle_tag = battle.battle_tag
            self._sw = {"my": 0}
            self._last_foes = {}
        for view in battle.opponents:
            sp = to_id(view.active.species) if view.active is not None else None
            last = self._last_foes.get(view.n)
            if last is not None and sp is not None and sp != last:
                self._sw[view.n] = self._sw.get(view.n, 0) + 1
            if sp is not None:
                self._last_foes[view.n] = sp
        foes = battle.live_opponents
        if not foes:
            return self.choose_random_move(battle)
        projs = {v.n: battle_to_state_ffa(battle, v.n, switch_counts=self._sw)
                 for v in foes}
        ev0 = None  # lazy fallback EV (untouched projections) for errors
        self._ctxs = {}
        self._rets = {}
        weights = self._weights(foes)
        rets = {v.n: self._retaliation(projs, v) for v in foes}
        options = []  # (name, ev, order[, move_id, target_n])
        force = battle.force_switch or not battle.available_moves
        if not force:
            move_opts = []
            for m in battle.available_moves:
                target = battle.move_request_target(m.id)
                rep = self._last_move.get(battle.battle_tag) == m.id
                if target is not None and needs_target(target):
                    for v in foes:
                        name = f"move:{m.id}@p{v.n + 1}"
                        try:
                            ev = self._ev_move_ffa(projs, foes, weights, rets,
                                                   m.id, v.n, repeat=rep)
                        except Exception as e:
                            if ev0 is None:
                                ev0 = sum(weights[w.n] * self._v(projs[w.n])
                                          for w in foes)
                            ev = ev0 - 2 * H.unmodelled_penalty
                            move_opts.append((name, ev, self.create_order(m),
                                              m.id, v.n))
                            if self._log:
                                self._log.write(json.dumps({
                                    "battle": battle.battle_tag,
                                    "turn": battle.turn,
                                    "error": f"{name}: {e!r}"}) + "\n")
                            continue
                        loc = target_loc(battle.my_side_n, v.n)
                        move_opts.append((name, ev,
                                          self.create_order(m, move_target=loc),
                                          m.id, v.n))
                else:
                    name = f"move:{m.id}"
                    try:
                        ev = self._ev_move_ffa(projs, foes, weights, rets,
                                               m.id, None, repeat=rep)
                    except Exception as e:
                        if ev0 is None:
                            ev0 = sum(weights[w.n] * self._v(projs[w.n])
                                      for w in foes)
                        ev = ev0 - 2 * H.unmodelled_penalty
                        move_opts.append((name, ev, self.create_order(m),
                                          m.id, None))
                        if self._log:
                            self._log.write(json.dumps({
                                "battle": battle.battle_tag,
                                "turn": battle.turn,
                                "error": f"{name}: {e!r}"}) + "\n")
                        continue
                    move_opts.append((name, ev, self.create_order(m),
                                      m.id, None))
            options += move_opts
            # terastallize candidates: re-evaluate top-3 damaging moves with
            # tera typing; picked only if they beat the best plain EV by a margin
            ap = battle.active_pokemon
            tera_type = None
            if battle.can_tera and ap is not None:
                if ap.tera_type is not None:
                    tera_type = ap.tera_type.name.title()
                else:
                    tera_type = T.tera_prior(to_id(ap.species))
            if tera_type and move_opts:
                plain_best = max(ev for _, ev, _, _, _ in move_opts)
                dmg = [o for o in move_opts
                       if T.MOVES.get(o[3], {}).get("category") != "status"]
                for name, _, _, move_id, target_n in sorted(
                        dmg, key=lambda x: x[1], reverse=True)[:3]:
                    try:
                        ev = self._ev_move_ffa(projs, foes, weights, rets,
                                               move_id, target_n,
                                               tera_type=tera_type)
                    except Exception:
                        continue
                    if ev > plain_best + H.tera_margin:
                        m = next(m for m in battle.available_moves
                                 if m.id == move_id)
                        loc = (target_loc(battle.my_side_n, target_n)
                               if target_n is not None else 0)
                        options.append((f"{name}|tera:{tera_type}", ev,
                                        self.create_order(m, terastallize=True,
                                                          move_target=loc),
                                        move_id, target_n))
        for s in battle.available_switches:
            name = f"switch:{to_id(s.species)}"
            try:
                ev = self._ev_switch_ffa(projs, foes, weights, s)
            except Exception as e:
                if ev0 is None:
                    ev0 = sum(weights[w.n] * self._v(projs[w.n]) for w in foes)
                ev = ev0 - 2 * H.unmodelled_penalty
                options.append((name, ev, self.create_order(s)))
                if self._log:
                    self._log.write(json.dumps({
                        "battle": battle.battle_tag, "turn": battle.turn,
                        "error": f"{name}: {e!r}"}) + "\n")
                continue
            options.append((name, ev, self.create_order(s)))
        if not options:
            return self.choose_random_move(battle)
        options.sort(key=lambda x: x[1], reverse=True)
        chosen = options[0][0]
        self._last_move[battle.battle_tag] = \
            options[0][3] if len(options[0]) > 3 else None
        if chosen.startswith("switch:"):
            self._sw["my"] += 1
        if self._log:
            self._log.write(json.dumps({
                "battle": battle.battle_tag, "turn": battle.turn,
                "chosen": chosen,
                "options": {n: round(e, 4) for n, e, *_ in options},
            }) + "\n")
            self._log.flush()
        return options[0][2]
