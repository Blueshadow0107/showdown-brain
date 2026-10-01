"""parser_doubles.py — gen 9 RANDOM DOUBLES replay protocol -> decision rows.

Same job as parser.py, but for `gen9randomdoublesbattle` logs where each side
fields TWO actives at position tags `p1a`/`p1b`/`p2a`/`p2b`.

ROW SCHEMA (top level, mirrors parser.py — one row per ORDER, i.e. per active
slot decision, NOT per joint pair):
    game        replay id
    turn        turn number the order belongs to
    player      "p1" / "p2" (acting side, like singles)
    slot        "p1a"/"p1b"/"p2a"/"p2b" — which active slot decided  [doubles-only]
    action_kind "move" | "switch"
    action_id   move id, or species id of the mon switched in;
                for |swap| orders: the position swapped into ("p1b")
    state       full PRE-TURN state dict (see below)
    outcome     1.0 / 0.0 / 0.5 — game result from the acting player's view
                (same as singles: winner name resolved at |win|, ties -> 0.5)

STATE DICT (keys identical to singles wherever semantics match):
    weather, terrain, trick_room   same as singles
    turn                           turn number the snapshot belongs to (v2)
    my / foe                       per side (acting player's perspective):
        active    LIST of 2 mon dicts, slot order [a, b]; None if the slot is
                  empty or its mon fainted and has not been replaced.
                  Each mon dict is singles' mon_public() plus "pos" — the
                  ABSOLUTE protocol position ("p1a".."p2b"), so rows from
                  either perspective stay cross-referable.
        fainted, remaining, bench_known, hazards, screens   same as singles
                  (screens values are TURN COUNTS, ticking down per turn — v2)
        tera_used, last_action   same as singles v2
        team      v2: mon_public_v2 of EVERY known mon (insertion order,
                  fainted included, active flagged via "pos")

TURN STRUCTURE IN THE LOG (verified against fixtures):
    |turn|N
    <voluntary switches, 0-2 per side>     <- player orders, executed pre-moves
    <|move| lines in speed order>          <- up to 4 (2 per side); a slot whose
                                              mon fainted orders nothing; `cant`
                                              replaces the line (no row, same as
                                              singles — the ordered move is not
                                              revealed)
    <self-switch lines: U-turn/Parting Shot/Baton Pass/etc.>  <- same slot as
                                              the move; suppressed by the
                                              one-row-per-slot guard
    end-of-turn effects, |upkeep|
    <replacement switches for mid-turn faints>  <- chosen for the upcoming turn;
                                              deferred to the NEXT |turn| snapshot
    |turn|N+1

DESIGN DECISIONS:
  * State is snapshotted at each |turn| (deep copy of both sides + globals) and
    every order of that turn is emitted against it: full PRE-TURN information,
    perspective-flipped per acting side. This differs from singles (which emits
    at first-move time) — doubles interleaves both players' moves, so there is
    no clean "player state moment" mid-turn.
  * One row per slot per turn max (`emitted` slot sets). A U-turn/Parting Shot
    move+switch pair is ONE order -> only the move row is emitted.
  * Replacement switches (logged post-|upkeep|, before the next |turn|) are the
    player's choice for the upcoming turn: they are buffered and emitted at the
    next |turn| against the FRESH snapshot — i.e. the state the player actually
    saw when choosing. Row turn number = the upcoming turn.
  * |swap|POS1|POS2 exchanges the two ally slots (Ally Switch). It emits a
    "switch" row with action_id = the position swapped into only if that slot
    has no row yet this turn (guards against double-counting if the log also
    carries the Ally Switch |move| line). No |swap| lines occur in the current
    fixture set, so this path is unexercised.
  * |replace| (Zoroark Illusion reveal) transfers the slot's state to the real
    nick and fixes the species — it is a reveal, not a decision: NO row. This
    intentionally differs from parser.py, which routes replace through h_switch
    and emits a spurious switch row.
  * |-singleturn| / |-singlemove| (Protect family, Follow Me, Helping Hand,
    Roost...) are stored as volatiles and cleared at |upkeep| (end of turn).
    -singlemove Glaive Rush instead clears on the user's next |move|.
  * |drag| = forced switch: state updated, no row (player did not choose).
  * fainted/remaining are derived from per-mon flags, not a counter: Revival
    Blessing revives make one mon faint twice (seen in fixtures), and a
    switch-in always clears the fainted flag.
  * Unknown events are skipped, not fatal — same contract as parser.py.
"""
import copy
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field

from sb.parser import Battle, Mon, parse_hp, strip_prefix, to_id

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "rows_v2_doubles.jsonl"

POS_RE = re.compile(r"^(p[12])([ab]?)$")


def split_pos(pos: str):
    """'p1a: Nick' -> ('p1', 'a', 'Nick'); 'p1: Name' -> ('p1', None, 'Name')."""
    head, _, nick = pos.partition(": ")
    m = POS_RE.match(head)
    if not m:
        return None
    return (m.group(1), m.group(2) or None, nick or head)


@dataclass
class DSide:
    """Doubles side: two active slots instead of one. The fainted count is
    derived from mon flags (not a counter) because Revival Blessing revives
    make a single mon faint twice."""
    pid: str = ""                       # "p1" / "p2" — needed for absolute pos tags
    name: str = ""
    teamsize: int = 6
    active: dict = field(default_factory=lambda: {"a": None, "b": None})
    mons: dict = field(default_factory=dict)  # nick -> Mon
    switches: int = 0                   # voluntary switches (momentum feature)
    hazards: dict = field(default_factory=dict)
    screens: dict = field(default_factory=dict)   # screen -> turns remaining (v2)
    tera_used: bool = False             # v2: side-wide, endgame-relevant
    last_action: str | None = None      # v2: previous decision ("move:x"/"switch:y")

    def mon(self, nick) -> Mon:
        return self.mons.setdefault(nick, Mon())

    @property
    def fainted(self) -> int:
        return sum(1 for m in self.mons.values() if m.fainted)


def flip(side: str) -> str:
    return "p2" if side == "p1" else "p1"


class DoublesBattle(Battle):
    """Subclass of sb.parser.Battle: mon-level `-event` handlers (damage,
    boosts, status, weather, hazards, item/ability...) are inherited unchanged;
    slot placement, turn scaffolding and emission are doubles-specific and
    overridden here."""

    def __init__(self, game_id: str):
        self.id = game_id
        self.sides = {"p1": DSide("p1"), "p2": DSide("p2")}
        self.weather = None
        self.terrain = None
        self.trick_room = False
        self.turn = 0
        self.rows = []
        self.winner = None
        self.tie = False
        # doubles bookkeeping
        self.snap = None                    # (sides_copy, weather, terrain, trick_room)
        self.emitted = {"p1": set(), "p2": set()}   # slots with a row this turn
        self.pending = []                   # buffered replacement switches
        self.post_upkeep = False
        self.turn_volatiles = []            # (side, nick, cond) cleared at upkeep

    # ---------- state -> row construction ----------
    def mon_public(self, m: Mon, pos: str) -> dict:
        d = {
            "species": m.species,
            "hp": round(m.hp, 4),
            "status": m.status,
            "boosts": {k: v for k, v in m.boosts.items() if v},
            "volatiles": sorted(m.volatiles),
            "item": m.item,
            "ability": m.ability,
            "moves": sorted(m.moves),
            "tera": m.tera,
            "pos": pos,
        }
        return d

    def mon_public_v2(self, m: Mon, pos: str | None, active: bool) -> dict:
        """Full-fidelity mon, doubles flavour: mon_public + v2 knowledge /
        usage / cadence fields. `pos` is the absolute slot for actives, None
        for bench mons; bench mons drop "volatiles" (active-only anyway)."""
        d = self.mon_public(m, pos)
        d.update({
            "fainted": m.fainted,
            "item_known": m.item_known,
            "ability_known": m.ability_known,
            "pp_used": dict(m.pp_used),
            "last_move": m.last_move,
            "consecutive": m.consecutive,
            "times_entered": m.times_entered,
        })
        if not active:
            d.pop("volatiles", None)
        return d

    def mon_pos(self, s: DSide, nick: str):
        """Absolute pos tag of the slot `nick` currently occupies, else None."""
        for sl in ("a", "b"):
            if s.active.get(sl) == nick:
                return f"{s.pid}{sl}"
        return None

    def side_public(self, s: DSide) -> dict:
        act = []
        for sl in ("a", "b"):
            nick = s.active.get(sl)
            m = s.mons.get(nick) if nick else None
            if m and not m.fainted:
                act.append(self.mon_public(m, f"{s.pid}{sl}"))
            else:
                act.append(None)
        bench = [n for n, m in s.mons.items()
                 if n not in (s.active["a"], s.active["b"]) and not m.fainted]
        return {
            "active": act,
            "fainted": s.fainted,
            "remaining": max(0, s.teamsize - s.fainted),
            "bench_known": len(bench),
            "switches": s.switches,
            "hazards": dict(s.hazards),
            "screens": dict(s.screens),
            # ---- state v2 ----
            "tera_used": s.tera_used,
            "last_action": s.last_action,
            "team": [self.mon_public_v2(m, self.mon_pos(s, n),
                                        self.mon_pos(s, n) is not None)
                     for n, m in s.mons.items()],
        }

    def emit(self, actor_side: str, slot: str, action_kind: str, action_id: str):
        if self.snap is None:
            return
        snap_sides, weather, terrain, trick_room = self.snap
        me = snap_sides[actor_side]
        foe = snap_sides[flip(actor_side)]
        state = {
            "turn": self.turn,
            "weather": weather,
            "terrain": terrain,
            "trick_room": trick_room,
            "my": self.side_public(me),
            "foe": self.side_public(foe),
        }
        # doubles snapshots are taken at |turn| (pre-decision by construction);
        # the state_pre key keeps the row schema uniform with the singles parser
        self.rows.append({
            "game": self.id,
            "turn": self.turn,
            "player": actor_side,
            "slot": actor_side + slot,
            "action_kind": action_kind,
            "action_id": action_id,
            "state": state,
            "state_pre": state,
        })
        # v2: record the action AFTER the row is built — rows of this turn all
        # carry the PREVIOUS turn's last action (the snapshot froze it)
        self.sides[actor_side].last_action = f"{action_kind}:{action_id}"

    def finish(self):
        for r in self.rows:
            if self.tie:
                r["outcome"] = 0.5
            else:
                r["outcome"] = 1.0 if r["player"] == self.winner else 0.0
        return self.rows

    # ---------- turn scaffolding ----------
    def h_turn(self, parts):
        self.turn = int(parts[2])
        # screens tick down at turn start, before the snapshot (mirror of
        # Battle.h_turn) so this turn's rows carry the post-decrement counts
        for side in self.sides.values():
            for sc in list(side.screens):
                side.screens[sc] -= 1
                if side.screens[sc] <= 0:
                    del side.screens[sc]
        self.snap = (copy.deepcopy(self.sides), self.weather, self.terrain,
                     self.trick_room)
        # replacement switches chosen after the previous upkeep belong to this
        # turn's decision phase; emit them against the fresh snapshot
        pend, self.pending = self.pending, []
        for s, sl, species in pend:
            self.emit(s, sl, "switch", species)
        self.emitted = {"p1": set(), "p2": set()}
        self.post_upkeep = False

    def h_upkeep(self, parts):
        for s, nick, cond in self.turn_volatiles:
            mon = self.sides[s].mons.get(nick)
            if mon:
                mon.volatiles.discard(cond)
        self.turn_volatiles = []
        self.post_upkeep = True

    # ---------- orders ----------
    def h_move(self, parts):
        pos, mv = parts[2], parts[3]
        sp = split_pos(pos)
        if not sp or not sp[1]:
            return
        s, sl, nick = sp
        side = self.sides[s]
        mon = side.mon(nick)
        if mon.species == "":
            mon.species = to_id(nick)
        mon.moves.add(to_id(mv))
        # v2: usage tracking — the per-turn snapshot was taken at |turn|, so
        # tracking updates here can never leak into this turn's rows
        m_id = to_id(mv)
        mon.pp_used[m_id] = mon.pp_used.get(m_id, 0) + 1
        mon.consecutive = mon.consecutive + 1 if mon.last_move == m_id else 0
        mon.last_move = m_id
        mon.volatiles.discard("mustrecharge")
        mon.volatiles.discard("glaiverush")
        if self.snap is not None and sl not in self.emitted[s]:
            self.emitted[s].add(sl)
            self.emit(s, sl, "move", m_id)

    def h_switch(self, parts, forced=False):
        pos, details = parts[2], parts[3]
        hp = parts[4] if len(parts) > 4 else ""
        sp = split_pos(pos)
        if not sp or not sp[1]:
            return
        s, sl, nick = sp
        side = self.sides[s]
        species = to_id(details.split(",")[0])
        old = side.active.get(sl)
        if old and old in side.mons and old != nick:
            self._apply_exit_effects(side.mons[old])
        if old and old in side.mons:
            side.mons[old].reset_on_switch()
        mon = side.mon(nick)
        mon.times_entered += 1  # any switch-in is an entry (drag included)
        mon.species = species
        mon.volatiles = set()
        mon.fainted = False  # fresh switch-in; also covers Revival Blessing revives
        hp_t = parse_hp(hp)
        if hp_t:
            cur, mx = hp_t
            if mx:
                mon.maxhp = mx
            mon.hp = cur / mon.maxhp if mon.maxhp else 0.0
        side.active[sl] = nick
        if not forced:
            side.switches += 1
        if forced or self.snap is None:
            return
        if self.post_upkeep:
            # replacement for a mid-turn faint: chosen for the upcoming turn
            self.pending.append((s, sl, species))
        elif sl not in self.emitted[s]:
            self.emitted[s].add(sl)
            self.emit(s, sl, "switch", species)

    def h_drag(self, parts):
        self.h_switch(parts, forced=True)

    def h_swap(self, parts):
        """|swap|p1a: Nick|p1b — ally position exchange (Ally Switch)."""
        sp = split_pos(parts[2])
        tgt = split_pos(parts[3]) if ": " in parts[3] else (None, None, parts[3])
        if not sp or not sp[1]:
            return
        s, sl, _ = sp
        side = self.sides[s]
        other = tgt[1] if tgt and tgt[1] else ("b" if sl == "a" else "a")
        side.active[sl], side.active[other] = side.active.get(other), side.active.get(sl)
        if self.snap is not None and sl not in self.emitted[s]:
            self.emitted[s].add(sl)
            self.emit(s, sl, "switch", f"{s}{other}")

    def h_replace(self, parts):
        """|replace|p2a: Zoroark|Zoroark, L84, M — Illusion broke: the slot's
        occupant is revealed to be a different mon. Transfer state, fix
        species, remap the slot. A reveal, not a decision: no row."""
        pos, details = parts[2], parts[3]
        sp = split_pos(pos)
        if not sp or not sp[1]:
            return
        s, sl, nick = sp
        side = self.sides[s]
        old = side.active.get(sl)
        mon = side.mon(nick)
        if old and old != nick and old in side.mons:
            m_old = side.mons.pop(old)
            mon.hp, mon.maxhp = m_old.hp, m_old.maxhp
            mon.status = m_old.status
            mon.boosts = m_old.boosts
            mon.volatiles = m_old.volatiles
            mon.item = m_old.item
            mon.ability = m_old.ability
            mon.moves |= m_old.moves
            mon.tera = m_old.tera
            mon.fainted = m_old.fainted
            # v2 tracking belongs to the mon, not the disguise — carry it over
            mon.item_known = m_old.item_known
            mon.ability_known = m_old.ability_known
            mon.pp_used = dict(m_old.pp_used)
            mon.last_move = m_old.last_move
            mon.consecutive = m_old.consecutive
            mon.times_entered = m_old.times_entered
        mon.species = to_id(details.split(",")[0])
        side.active[sl] = nick
        # h_replace: an illusion reveal, not a decision — no switch counted,
        # no entry/exit effects (the same mon was on the field all along)

    # ---------- doubles-specific mon events ----------
    def h_singleturn(self, parts):
        pos, cond = parts[2], to_id(strip_prefix(parts[3]))
        sp = split_pos(pos)
        if not sp:
            return
        s, _, nick = sp
        self.sides[s].mon(nick).volatiles.add(cond)
        self.turn_volatiles.append((s, nick, cond))

    def h_singlemove(self, parts):
        """Glaive Rush-style: lasts until the user's next move, not end of turn."""
        pos, cond = parts[2], to_id(strip_prefix(parts[3]))
        sp = split_pos(pos)
        if not sp:
            return
        self.sides[sp[0]].mon(sp[2]).volatiles.add(cond)

    def h_mustrecharge(self, parts):
        pos = parts[2]
        sp = split_pos(pos)
        if sp:
            self.sides[sp[0]].mon(sp[2]).volatiles.add("mustrecharge")

    def h_faint(self, parts):
        pos = parts[2]
        sp = split_pos(pos)
        if not sp:
            return
        s, sl, nick = sp
        side = self.sides[s]
        mon = side.mon(nick)
        mon.fainted = True   # idempotent — the count derives from flags
        mon.hp = 0.0
        mon.status = None
        if sl and side.active.get(sl) == nick:
            side.active[sl] = None

    def h_clearnegativeboost(self, parts):
        pos = parts[2]
        sp = split_pos(pos)
        if not sp:
            return
        mon = self.sides[sp[0]].mon(sp[2])
        mon.boosts = {k: (0 if v < 0 else v) for k, v in mon.boosts.items()}

    def h_formechange(self, parts):
        pos, forme = parts[2], parts[3]
        sp = split_pos(pos)
        if sp:
            self.sides[sp[0]].mon(sp[2]).species = to_id(forme.split(",")[0])

    def h_transform(self, parts):
        """Copy the target's species + revealed moves (Ditto/Mew)."""
        sp, tp = split_pos(parts[2]), split_pos(parts[3])
        if not sp or not tp:
            return
        mon = self.sides[sp[0]].mon(sp[2])
        tgt = self.sides[tp[0]].mons.get(tp[2])
        if tgt:
            mon.species = tgt.species
            mon.moves |= tgt.moves
        mon.volatiles.add("transformed")


HANDLERS = {
    "player": "h_player", "teamsize": "h_teamsize", "turn": "h_turn",
    "upkeep": "h_upkeep",
    "switch": "h_switch", "move": "h_move", "swap": "h_swap",
    "replace": "h_replace",
    "drag": "h_drag",
    "-damage": "h_damage", "-heal": "h_heal", "-sethp": "h_damage",
    "-status": "h_status", "-curestatus": "h_curestatus",
    "-boost": "h_boost", "-unboost": lambda self, p: self.h_boost(p, sign=-1),
    "-setboost": "h_setboost", "-clearboost": "h_clearboost",
    "-clearallboost": "h_clearboost", "-clearnegativeboost": "h_clearnegativeboost",
    "-start": "h_start", "-end": "h_end",
    "-weather": "h_weather", "-fieldstart": "h_fieldstart", "-fieldend": "h_fieldend",
    "-sidestart": "h_sidestart", "-sideend": "h_sideend",
    "-ability": "h_ability", "-item": "h_item", "-enditem": "h_enditem",
    "-terastallize": "h_terastallize", "-mega": "h_detailschange",
    "-primal": "h_detailschange",
    "-singleturn": "h_singleturn", "-singlemove": "h_singlemove",
    "-mustrecharge": "h_mustrecharge",
    "-formechange": "h_formechange", "-transform": "h_transform",
    "detailschange": "h_detailschange",
    "faint": "h_faint",   # NB: protocol event is |faint| (no dash) — parser.py's
                          # "-faint" key never fires; don't copy that
    "win": "h_win", "tie": "h_tie",
}

# events deliberately ignored (no state change or not decision-relevant):
#   cant -miss -fail -immune -supereffective -resisted -crit -activate -block
#   -prepare -hitcount -anim -hint -message -c -j -l -t: -raw -inactive -rule
#   -gametype -gen -tier -rated -start -error -n


def parse_log(text: str, game_id: str, seen: Counter | None = None):
    """Parse one doubles replay log -> list of rows ([] if no winner/tie)."""
    b = DoublesBattle(game_id)
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        h = HANDLERS.get(parts[1])
        if not h:
            if seen is not None and parts[1].startswith("-"):
                seen[parts[1]] += 1
            continue
        try:
            if callable(h):
                h(b, parts)
            else:
                getattr(b, h)(parts)
        except Exception:
            continue  # malformed edge line — keep the battle going
    if not (b.winner or b.tie):
        return []
    return b.finish()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    logs = sorted(RAW.glob("gen9randomdoublesbattle-*.log"))
    n_rows = 0
    n_games = 0
    kinds = Counter()
    seen = Counter()
    with OUT.open("w") as f:
        for i, path in enumerate(logs):
            rows = parse_log(path.read_text(errors="replace"), path.stem, seen)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n_rows += len(rows)
            n_games += bool(rows)
            kinds.update(r["action_kind"] for r in rows)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(logs)} logs -> {n_rows} rows", flush=True)
    print(f"done: {n_games}/{len(logs)} games -> {n_rows} rows at {OUT}")
    print(f"action kinds: {dict(kinds)}")
    print(f"rows/game: {n_rows / n_games:.2f}" if n_games else "no games")
    print(f"unhandled '-' events: {dict(seen)}")


if __name__ == "__main__":
    main()
