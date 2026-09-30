"""parser.py — showdown replay protocol -> battle state -> decision rows.

Replays are line protocol. We track full battle state per side (from the acting
player's perspective at each decision point) and emit one row per decision:
(state, action, outcome). Unknown/unhandled events are skipped, not fatal.
"""
import json
import pathlib
import re
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "rows.jsonl"

STATUSES = {"brn", "par", "psn", "tox", "slp", "frz"}
BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def parse_hp(tok: str):
    """'123/456' | '0 fnt' | '100/100' -> (cur, max) or None."""
    tok = tok.strip()
    if not tok:
        return None
    if tok.endswith(" fnt"):
        return (0.0, None)
    m = re.match(r"(\d+(?:\.\d+)?)/(\d+)", tok)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


def strip_prefix(cond: str) -> str:
    for p in ("move: ", "ability: ", "item: "):
        if cond.startswith(p):
            cond = cond[len(p):]
    return to_id(cond)


@dataclass
class Mon:
    species: str = ""
    hp: float = 1.0
    maxhp: float = 100.0
    status: str | None = None
    boosts: dict = field(default_factory=lambda: {s: 0 for s in BOOST_STATS})
    volatiles: set = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    moves: set = field(default_factory=set)
    tera: str | None = None
    fainted: bool = False

    def reset_on_switch(self):
        self.boosts = {s: 0 for s in BOOST_STATS}
        self.volatiles = set()


@dataclass
class Side:
    name: str = ""                      # player name (from |player|)
    teamsize: int = 6
    active: str | None = None           # nickname of active mon
    mons: dict = field(default_factory=dict)  # nick -> Mon
    fainted: int = 0
    hazards: dict = field(default_factory=dict)   # cond -> count (spikes) or 1
    screens: dict = field(default_factory=dict)   # cond -> 1

    def mon(self, nick) -> Mon:
        return self.mons.setdefault(nick, Mon())


def side_of(pos: str) -> str:
    return "p1" if pos.startswith("p1") else "p2"


class Battle:
    def __init__(self, game_id: str):
        self.id = game_id
        self.sides = {"p1": Side(), "p2": Side()}
        self.weather = None
        self.terrain = None
        self.trick_room = False
        self.turn = 0
        self.emitted = {"p1": False, "p2": False}
        self.rows = []
        self.winner = None
        self.tie = False

    # ---------- row construction ----------
    def mon_public(self, m: Mon) -> dict:
        return {
            "species": m.species,
            "hp": round(m.hp, 4),
            "status": m.status,
            "boosts": {k: v for k, v in m.boosts.items() if v},
            "volatiles": sorted(m.volatiles),
            "item": m.item,
            "ability": m.ability,
            "moves": sorted(m.moves),
            "tera": m.tera,
        }

    def side_public(self, s: Side) -> dict:
        bench = [n for n, m in s.mons.items() if n != s.active and not m.fainted]
        return {
            "active": self.mon_public(s.mons[s.active]) if s.active in s.mons else None,
            "fainted": s.fainted,
            "remaining": max(0, s.teamsize - s.fainted),
            "bench_known": len(bench),
            "hazards": dict(s.hazards),
            "screens": dict(s.screens),
        }

    def emit(self, actor_side: str, action_kind: str, action_id: str):
        me = self.sides[actor_side]
        foe_side = "p2" if actor_side == "p1" else "p1"
        foe = self.sides[foe_side]
        self.rows.append({
            "game": self.id,
            "turn": self.turn,
            "player": actor_side,
            "action_kind": action_kind,
            "action_id": action_id,
            "state": {
                "weather": self.weather,
                "terrain": self.terrain,
                "trick_room": self.trick_room,
                "my": self.side_public(me),
                "foe": self.side_public(foe),
            },
        })

    def finish(self):
        for r in self.rows:
            if self.tie:
                r["outcome"] = 0.5
            else:
                r["outcome"] = 1.0 if r["player"] == self.winner else 0.0
        return self.rows

    # ---------- event handlers ----------
    def h_switch(self, parts, forced=False):
        pos, details, hp = parts[2], parts[3], parts[4] if len(parts) > 4 else ""
        side = self.sides[side_of(pos)]
        nick = pos.split(": ", 1)[1] if ": " in pos else pos
        species = to_id(details.split(",")[0])
        mon = side.mon(nick)
        if mon.fainted:  # revived (Revival Blessing) — no longer fainted
            mon.fainted = False
            side.fainted = max(0, side.fainted - 1)
        if side.active and side.active in side.mons:
            side.mons[side.active].reset_on_switch()
        side.active = nick
        mon.species = species
        mon.volatiles = set()
        hp_t = parse_hp(hp)
        if hp_t:
            cur, mx = hp_t
            if mx:
                mon.maxhp = mx
            mon.hp = cur / mon.maxhp if mon.maxhp else 0.0
        if not forced and self.turn >= 1:
            self.emit(side_of(pos), "switch", species)

    def h_move(self, parts):
        pos, mv = parts[2], parts[3]
        s = side_of(pos)
        side = self.sides[s]
        nick = pos.split(": ", 1)[1]
        mon = side.mon(nick)
        if mon.species == "":
            mon.species = to_id(nick)
        mon.moves.add(to_id(mv))
        if not self.emitted[s]:
            self.emitted[s] = True
            self.emit(s, "move", to_id(mv))

    def h_damage(self, parts):
        pos, hp = parts[2], parts[3]
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        hp_t = parse_hp(hp)
        if hp_t:
            cur, mx = hp_t
            if mx:
                mon.maxhp = mx
            mon.hp = cur / mon.maxhp if mon.maxhp else 0.0
        if "[from] " in parts[-1] and "recoil" not in parts[-1]:
            pass  # source tags; hp already applied

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
        pos, cond = parts[2], strip_prefix(parts[3])
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).volatiles.add(cond)

    def h_end(self, parts):
        pos = parts[2]
        cond = strip_prefix(parts[3]) if len(parts) > 3 else None
        mon = self.sides[side_of(pos)].mon(pos.split(": ", 1)[1])
        if cond:
            mon.volatiles.discard(cond)
        else:
            mon.volatiles = set()

    def h_weather(self, parts):
        w = to_id(parts[2])
        self.weather = None if w == "none" else w

    def h_fieldstart(self, parts):
        f = strip_prefix(parts[2])
        if f == "trickroom":
            self.trick_room = True
        else:
            self.terrain = f

    def h_fieldend(self, parts):
        f = strip_prefix(parts[2])
        if f == "trickroom":
            self.trick_room = False
        elif f == self.terrain:
            self.terrain = None

    def h_sidestart(self, parts):
        side = self.sides["p1" if parts[2].startswith("p1") else "p2"]
        cond = strip_prefix(parts[3])
        side.hazards[cond] = side.hazards.get(cond, 0) + 1

    def h_sideend(self, parts):
        side = self.sides["p1" if parts[2].startswith("p1") else "p2"]
        cond = strip_prefix(parts[3])
        side.hazards.pop(cond, None)

    def h_ability(self, parts):
        pos, ab = parts[2], to_id(parts[3])
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).ability = ab

    def h_item(self, parts):
        pos, it = parts[2], to_id(parts[3])
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).item = it

    def h_enditem(self, parts):
        pos = parts[2]
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).item = None

    def h_terastallize(self, parts):
        pos, t = parts[2], parts[3].strip().title()
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).tera = t

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
        self.sides[side_of(pos)].mon(pos.split(": ", 1)[1]).species = to_id(details.split(",")[0])

    def h_player(self, parts):
        self.sides[parts[2]].name = parts[3]

    def h_teamsize(self, parts):
        self.sides[parts[2]].teamsize = int(parts[3])

    def h_turn(self, parts):
        self.turn = int(parts[2])
        self.emitted = {"p1": False, "p2": False}

    def h_win(self, parts):
        name = parts[2]
        for s, side in self.sides.items():
            if side.name == name:
                self.winner = s

    def h_tie(self, parts):
        self.tie = True


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
    "-weather": "h_weather", "-fieldstart": "h_fieldstart", "-fieldend": "h_fieldend",
    "-sidestart": "h_sidestart", "-sideend": "h_sideend",
    "-ability": "h_ability", "-item": "h_item", "-enditem": "h_enditem",
    "-terastallize": "h_terastallize", "-mega": "h_detailschange",
    "-primal": "h_detailschange", "faint": "h_faint",
    "detailschange": "h_detailschange", "win": "h_win", "tie": "h_tie",
}


def parse_log(text: str, game_id: str):
    b = Battle(game_id)
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        h = HANDLERS.get(parts[1])
        if not h:
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
    logs = sorted(RAW.glob("gen9randombattle-*.log"))  # not the doubles logs
    n_rows = 0
    with OUT.open("w") as f:
        for i, path in enumerate(logs):
            rows = parse_log(path.read_text(errors="replace"), path.stem)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n_rows += len(rows)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(logs)} logs -> {n_rows} rows", flush=True)
    print(f"done: {len(logs)} logs -> {n_rows} rows at {OUT}")


if __name__ == "__main__":
    main()
