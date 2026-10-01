"""showdown_state.py — poke-env Battle -> parser-shaped state dict (for V(s))."""
from poke_env.battle import Battle

from sb.features import to_id

WEATHER_MAP = {"rain": "raindance", "sunnyday": "sunnyday", "desolateland": "sunnyday",
               "sandstorm": "sandstorm", "snow": "snow", "hail": "snow"}
TERRAIN_MAP = {"electricterrain": "electricterrain", "grassyterrain": "grassyterrain",
               "mistyterrain": "mistyterrain", "psychicterrain": "psychicterrain"}
HAZARD_MAP = {"stealthrock": "stealthrock", "spikes": "spikes", "toxicspikes": "toxicspikes",
              "stickyweb": "stickyweb", "reflect": "reflect", "lightscreen": "lightscreen",
              "auroraveil": "auroraveil", "tailwind": "tailwind"}


def _mon_public(m) -> dict:
    return {
        "species": to_id(m.species),
        "level": m.level,
        "hp": round(float(m.current_hp_fraction), 4),
        "status": m.status.name.lower() if m.status else None,
        "boosts": {k: v for k, v in m.boosts.items() if v},
        "volatiles": [],
        "item": to_id(m.item) if m.item else None,
        "ability": to_id(m.ability) if m.ability else None,
        "moves": sorted(m.moves.keys()),
        "tera": m.tera_type.name.title() if m.is_terastallized and m.tera_type else None,
    }


def _mon_public_v2(m, active: bool) -> dict:
    """Full-fidelity live mon matching parser.mon_public_v2. Knowledge flags:
    for our team everything is known; for the foe, poke-env leaves unrevealed
    attrs None — known := is not None (a consumed item reads as unknown-live,
    a documented approximation). last_move/consecutive/times_entered are not
    exposed by poke-env: null here (the agent keeps its own counters)."""
    d = _mon_public(m)
    d.update({
        "fainted": bool(m.fainted),
        "item_known": m.item is not None,
        "ability_known": m.ability is not None,
        "pp_used": {mv.id: max(0, mv.max_pp - mv.current_pp)
                    for mv in m.moves.values() if mv.current_pp < mv.max_pp},
        "last_move": None,
        "consecutive": None,
        "times_entered": None,
    })
    if not active:
        d.pop("volatiles", None)
    return d


SCREENS = ("reflect", "lightscreen", "auroraveil", "tailwind")


def _side_public(team, active, conditions, switches: int = 0,
                 tera_used: bool = False, last_action=None) -> dict:
    mons = list(team.values())
    fainted = sum(1 for m in mons if m.fainted or m.current_hp_fraction == 0)
    hazards, screens = {}, {}
    for cond, n in conditions.items():
        key = HAZARD_MAP.get(_key(cond))
        if key:
            if key in SCREENS:
                screens[key] = n if isinstance(n, int) else 1
            else:
                hazards[key] = max(hazards.get(key, 0), n if isinstance(n, int) else 1)
    return {
        "active": _mon_public(active) if active else None,
        "fainted": fainted,
        "remaining": max(0, len(mons) - fainted) if mons else 6 - fainted,
        "bench_known": len(mons) - 1,
        "switches": switches,
        "hazards": hazards,
        "screens": screens,
        "tera_used": tera_used,
        "last_action": last_action,
        "team": [_mon_public_v2(m, m is active) for m in mons],
    }


def _key(obj):
    return obj.name.lower() if hasattr(obj, "name") else str(obj).split(".")[-1].lower()


def battle_to_state(battle: Battle, switch_counts: dict | None = None,
                    last_actions: dict | None = None) -> dict:
    """State from the agent player's perspective (matches sb.parser rows).
    switch_counts: optional {"my": n, "foe": n} momentum counters maintained
    by the agent (parser rows carry their own). last_actions: optional
    {"my": str|None, "foe": str|None} — the sides' previous decisions."""
    sw = switch_counts or {}
    la = last_actions or {}
    weather = None
    if battle.weather:
        weather = WEATHER_MAP.get(_key(next(iter(battle.weather))), "snow")
    terrain, trick_room = None, False
    for f in battle.fields:
        key = _key(f)
        if key == "trickroom":
            trick_room = True
        elif key in TERRAIN_MAP:
            terrain = TERRAIN_MAP[key]
    return {
        "turn": battle.turn,
        "weather": weather,
        "terrain": terrain,
        "trick_room": trick_room,
        "my": _side_public(battle.team, battle.active_pokemon,
                           battle.side_conditions, sw.get("my", 0),
                           tera_used=bool(getattr(battle, "_used_tera", False)),
                           last_action=la.get("my")),
        "foe": _side_public(battle.opponent_team, battle.opponent_active_pokemon,
                            battle.opponent_side_conditions, sw.get("foe", 0),
                            tera_used=bool(getattr(battle, "_opponent_used_tera", False)),
                            last_action=la.get("foe")),
    }
