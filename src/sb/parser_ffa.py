"""parser_ffa.py — gen 9 FREE-FOR-ALL replay protocol -> pairwise v2 rows.

Same job as parser.py, but for `gen9freeforallrandombattle` logs: four sides
(p1..p4), one active slot each at positions `p1a`..`p4b` (the slot LETTER
varies by side — observed p1a/p2a/p3b/p4b — but a side keeps its slot all
game). Every decision is projected into one v2 row PER LIVE OPPONENT, so the
singles featurizer and V model consume FFA rows unchanged.

ROW SCHEMA (top level — parser.py's keys plus ONE addition):
    game        replay id
    turn        turn number
    player      "p1".."p4" — acting side
    foe_n       INT 0..3 — index of the opponent this row is projected
                against (p1->0, ..., p4->3); the only key parser.py lacks
    action_kind "move" | "switch"
    action_id   move id, or species id of the mon switched in
    state       post-decision pairwise state (my = actor, foe = foe_n's side)
    state_pre   pre-decision pairwise state (same my/foe assignment)
    outcome     1.0 if `player` wins the whole FFA (|win| names them),
                0.0 otherwise; 0.5 on |tie|

STATE DICT: identical to parser.py's v2 state — {"turn","weather","terrain",
"trick_room","my","foe"} with the per-side v2 fields (team[] with knowledge
flags, tera_used, screens, last_action, ...). Only the row's perspective
differs: `foe` is one chosen opponent, not "the" opponent.

PROTOCOL FACTS (verified against real FFA replay logs):
  * `|teamsize|pN|6` — FFA random teams are SIX mons; the log's teamsize is
    correct, so elimination = fainted count >= teamsize, exactly like singles.
  * `|-sidestart|pN: username|...` / `|-sideend|` key the side as "pN: name",
    not a bare pN — the side id is always the first two chars.
  * `|player|pN|` (empty name) marks a disconnect; it must NOT clobber the
    side's name or `|win|` resolution breaks. Non-empty lines only.
  * Eliminated players (all mons fainted) receive `wait` requests and produce
    no further decisions — they simply stop appearing, so no rows. Rows are
    only projected against foes with fainted < teamsize AT DECISION TIME,
    which also means < 3 foe rows once anyone is eliminated.
  * Teampreview: no rows (parser.py skips preview; FFA mirrors it).
  * Silent switch-out mechanics (Regenerator heal, Natural Cure) are mirrored
    via the inherited _apply_exit_effects, same as singles.
"""
import json
import pathlib
from collections import Counter

from sb.parser import (
    BOOST_STATS, SCREEN_TURNS, STATUSES, Battle, Side, parse_hp, strip_prefix,
    to_id,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "rows_ffa.jsonl"

SIDES = ("p1", "p2", "p3", "p4")


def side_of(pos: str) -> str:
    """'p3b: Nick' -> 'p3' — FFA positions carry one slot letter; the side is
    the first two chars (also covers 'p3: name' side-condition keys)."""
    return pos[:2]


class FFABattle(Battle):
    """parser.Battle with four sides and pairwise row emission. parser.py's
    mon-level handlers resolve sides through a p1/p2-only side_of, so the FFA
    handlers below are re-creations of them with 4-side resolution (the same
    reason parser_doubles re-creates its handlers instead of inheriting). The
    state builders (mon_public/side_public), exit effects and finish() are
    inherited unchanged — Side and the state shape are identical to singles."""

    def __init__(self, game_id: str):
        self.id = game_id
        self.sides = {s: Side() for s in SIDES}
        self.weather = None
        self.terrain = None
        self.trick_room = False
        self.turn = 0
        self.emitted = {s: False for s in SIDES}
        self.rows = []
        self.winner = None
        self.tie = False

    # ---------- row construction ----------
    def live_foes(self, actor_side: str) -> list[str]:
        """Sides still in the game besides the actor (pairwise targets)."""
        return [s for s in SIDES
                if s != actor_side
                and self.sides[s].fainted < self.sides[s].teamsize]

    def _row_state(self, actor_side: str, foe_side: str) -> dict:
        return {
            "turn": self.turn,
            "weather": self.weather,
            "terrain": self.terrain,
            "trick_room": self.trick_room,
            "my": self.side_public(self.sides[actor_side]),
            "foe": self.side_public(self.sides[foe_side]),
        }

    def _pre_states(self, actor_side: str) -> dict:
        """Pre-decision state per live foe — the pairwise flavour of
        parser.py's `pre = self._row_state(actor)`."""
        return {f: self._row_state(actor_side, f) for f in self.live_foes(actor_side)}

    def emit(self, actor_side: str, action_kind: str, action_id: str,
             pre: dict | None = None):
        me = self.sides[actor_side]
        for foe_side in self.live_foes(actor_side):
            row = {
                "game": self.id,
                "turn": self.turn,
                "player": actor_side,
                "foe_n": int(foe_side[1]) - 1,
                "action_kind": action_kind,
                "action_id": action_id,
                "state": self._row_state(actor_side, foe_side),
            }
            if pre is not None:
                row["state_pre"] = pre[foe_side]
            self.rows.append(row)
        me.last_action = f"{action_kind}:{action_id}"

    # ---------- event handlers (parser.py bodies, 4-side side_of) ----------
    def h_player(self, parts):
        # `|player|pN|` (empty name) marks a disconnect — keep the old name
        # or `|win|` resolution breaks for players who drop mid-game
        if len(parts) > 3 and parts[3]:
            self.sides[parts[2]].name = parts[3]

    def h_turn(self, parts):
        self.turn = int(parts[2])
        self.emitted = {s: False for s in SIDES}
        for side in self.sides.values():
            for sc in list(side.screens):
                side.screens[sc] -= 1
                if side.screens[sc] <= 0:
                    del side.screens[sc]

    def h_switch(self, parts, forced=False):
        pos, details, hp = parts[2], parts[3], parts[4] if len(parts) > 4 else ""
        actor = side_of(pos)
        side = self.sides[actor]
        nick = pos.split(": ", 1)[1] if ": " in pos else pos
        species = to_id(details.split(",")[0])
        pre = self._pre_states(actor) if not forced and self.turn >= 1 else None
        mon = side.mon(nick)
        if mon.fainted:  # revived (Revival Blessing) — no longer fainted
            mon.fainted = False
            side.fainted = max(0, side.fainted - 1)
        if side.active and side.active in side.mons and side.active != nick:
            self._apply_exit_effects(side.mons[side.active])
        if side.active and side.active in side.mons:
            side.mons[side.active].reset_on_switch()
        side.active = nick
        mon.times_entered += 1
        mon.species = species
        mon.volatiles = set()
        hp_t = parse_hp(hp)
        if hp_t:
            cur, mx = hp_t
            if mx:
                mon.maxhp = mx
            mon.hp = cur / mon.maxhp if mon.maxhp else 0.0
        if not forced and self.turn >= 1:
            side.switches += 1
            self.emit(actor, "switch", species, pre=pre)

    def h_move(self, parts):
        pos, mv = parts[2], parts[3]
        s = side_of(pos)
        side = self.sides[s]
        nick = pos.split(": ", 1)[1]
        mon = side.mon(nick)
        if mon.species == "":
            mon.species = to_id(nick)
        m_id = to_id(mv)
        pre = None
        if not self.emitted[s]:
            pre = self._pre_states(s)
        # v2: usage tracking (after the pre-decision snapshot)
        mon.moves.add(m_id)
        mon.pp_used[m_id] = mon.pp_used.get(m_id, 0) + 1
        mon.consecutive = mon.consecutive + 1 if mon.last_move == m_id else 0
        mon.last_move = m_id
        if pre is not None:
            self.emitted[s] = True
            self.emit(s, "move", m_id, pre=pre)

    def h_damage(self, parts):
        pos, hp = parts[2], parts[3]
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        hp_t = parse_hp(hp)
        if hp_t:
            cur, mx = hp_t
            if mx:
                mon.maxhp = mx
            mon.hp = cur / mon.maxhp if mon.maxhp else 0.0

    def h_heal(self, parts):
        self.h_damage(parts)

    def h_status(self, parts):
        pos, st = parts[2], to_id(parts[3])
        if st in STATUSES:
            self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).status = st

    def h_curestatus(self, parts):
        self.sides[side_of(parts[2])].mon(parts[2].split(": ", 1)[1]).status = None

    def h_boost(self, parts, sign=1):
        pos, stat, n = parts[2], to_id(parts[3]), int(parts[4])
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        if stat in mon.boosts:
            mon.boosts[stat] = max(-6, min(6, mon.boosts[stat] + sign * n))

    def h_setboost(self, parts):
        pos, stat, n = parts[2], to_id(parts[3]), int(parts[4])
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        if stat in mon.boosts:
            mon.boosts[stat] = max(-6, min(6, n))

    def h_clearboost(self, parts):
        self.sides[side_of(parts[2])].mon(parts[2].split(": ", 1)[1]).boosts = {
            s: 0 for s in BOOST_STATS}

    def h_start(self, parts):
        pos, cond = parts[2], to_id(strip_prefix(parts[3]))
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).volatiles.add(cond)

    def h_end(self, parts):
        pos = parts[2]
        cond = to_id(strip_prefix(parts[3])) if len(parts) > 3 else None
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        if cond:
            mon.volatiles.discard(cond)
        else:
            mon.volatiles = set()

    def h_sidestart(self, parts):
        side = self.sides[parts[2][:2]]
        cond = strip_prefix(parts[3])
        if cond in SCREEN_TURNS:
            side.screens[cond] = SCREEN_TURNS[cond]  # v2: timed, decremented per turn
        else:
            side.hazards[cond] = side.hazards.get(cond, 0) + 1

    def h_sideend(self, parts):
        side = self.sides[parts[2][:2]]
        cond = strip_prefix(parts[3])
        if cond in SCREEN_TURNS:
            side.screens.pop(cond, None)
        else:
            side.hazards.pop(cond, None)

    def h_ability(self, parts):
        pos, ab = parts[2], to_id(parts[3])
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        mon.ability = ab
        mon.ability_known = True

    def h_item(self, parts):
        pos, it = parts[2], to_id(parts[3])
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        mon.item = it
        mon.item_known = True

    def h_enditem(self, parts):
        pos = parts[2]
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        mon.item = None
        mon.item_known = True  # consumed/removed = revealed

    def h_terastallize(self, parts):
        pos, t = parts[2], parts[3].strip().title()
        side = self.sides[side_of(pos)]
        side.mon(pos.split(": ", 1)[1]).tera = t
        side.tera_used = True

    def h_faint(self, parts):
        pos = parts[2]
        side = self.sides[side_of(pos)]
        mon = side.mon(pos.split(": ", 1)[1])
        mon.fainted = True
        mon.hp = 0.0
        mon.status = None
        side.fainted += 1

    def h_detailschange(self, parts):
        pos, details = parts[2], parts[3]
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).species = to_id(
            details.split(",")[0])


HANDLERS = {
    "player": "h_player", "teamsize": "h_teamsize", "turn": "h_turn",
    "switch": "h_switch", "move": "h_move",
    "drag": lambda self, p: self.h_switch(p, forced=True),
    "replace": "h_switch",
    "-damage": "h_damage", "-heal": "h_heal", "-sethp": "h_damage",
    "-status": "h_status", "-curestatus": "h_curestatus",
    "-boost": "h_boost", "-unboost": lambda self, p: self.h_boost(p, sign=-1),
    "-setboost": "h_setboost", "-clearboost": "h_clearboost",
    "-clearallboost": "h_clearboost",
    "-start": "h_start", "-end": "h_end",
    "-weather": "h_weather", "-fieldstart": "h_fieldstart",
    "-fieldend": "h_fieldend",
    "-sidestart": "h_sidestart", "-sideend": "h_sideend",
    "-ability": "h_ability", "-item": "h_item", "-enditem": "h_enditem",
    "-terastallize": "h_terastallize", "-mega": "h_detailschange",
    "-primal": "h_detailschange", "faint": "h_faint",
    "detailschange": "h_detailschange", "win": "h_win", "tie": "h_tie",
}
# h_teamsize / h_win / h_tie / h_weather / h_fieldstart / h_fieldend resolve
# through inheritance — they touch self.sides generically and no positions.


def parse_log(text: str, game_id: str, seen: Counter | None = None,
              skips: Counter | None = None):
    """Parse one FFA replay log -> list of rows ([] if no winner/tie).
    `seen` counts unhandled '-' protocol events (protocol surface discovery,
    same as parser_doubles); `skips` counts lines whose handler raised —
    malformed edge lines are skipped, never fatal."""
    b = FFABattle(game_id)
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
            if skips is not None:
                skips[parts[1]] += 1
            continue  # malformed edge line — keep the battle going
    if not (b.winner or b.tie):
        return []
    return b.finish()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    logs = sorted(RAW.glob("gen9freeforallrandombattle-*.log"))  # FFA only
    n_rows = 0
    n_games = 0
    kinds = Counter()
    seen = Counter()
    with OUT.open("w") as f:
        for i, path in enumerate(logs):
            skips = Counter()
            rows = parse_log(path.read_text(errors="replace"), path.stem,
                             seen, skips)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n_rows += len(rows)
            n_games += bool(rows)
            kinds.update(r["action_kind"] for r in rows)
            if skips:
                print(f"skips in {path.stem}: {dict(skips)}")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(logs)} logs -> {n_rows} rows", flush=True)
    print(f"done: {n_games}/{len(logs)} games -> {n_rows} rows at {OUT}")
    print(f"action kinds: {dict(kinds)}")
    if n_games:
        print(f"rows/game: {n_rows / n_games:.2f}")
    print(f"unhandled '-' events: {dict(seen)}")


if __name__ == "__main__":
    main()
