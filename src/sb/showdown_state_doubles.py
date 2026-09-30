"""showdown_state_doubles.py — poke_env DoubleBattle -> parser_doubles state dict.

Same job as showdown_state.py but each side fields TWO active slots:
`active` is a list of 2 mon dicts (None for empty/fainted slots), and each
mon dict carries "pos" — the absolute protocol position ("p1a".."p2b"), so
rows/states from either perspective stay cross-referable with parser_doubles
output. Featurize via sb.features_doubles.state_features_d(state, slot).
"""
from poke_env.battle import DoubleBattle

from sb.features import to_id
from sb.showdown_state import HAZARD_MAP, TERRAIN_MAP, WEATHER_MAP, _key


def _mon_public_d(m, pos: str) -> dict:
    return {
        "species": to_id(m.species),
        "level": m.level,
        "hp": round(float(m.current_hp_fraction), 4),
        "status": m.status.name.lower() if m.status else None,
        "boosts": {k: v for k, v in m.boosts.items() if v},
        # poke-env tracks all active effects; the doubles VOL_D feature set
        # picks out the competitive-relevant ones
        "volatiles": sorted(to_id(e.name) for e in m.effects),
        "item": to_id(m.item) if m.item else None,
        "ability": to_id(m.ability) if m.ability else None,
        "moves": sorted(m.moves.keys()),
        "tera": m.tera_type.name.title() if m.is_terastallized and m.tera_type else None,
        "pos": pos,
    }


def _side_public_d(team, actives, conditions, pid: str) -> dict:
    mons = list(team.values())
    fainted = sum(1 for m in mons if m.fainted or m.current_hp_fraction == 0)
    act = []
    for i, m in enumerate(actives):
        if m is not None and not m.fainted:
            act.append(_mon_public_d(m, f"{pid}{'ab'[i]}"))
        else:
            act.append(None)
    bench = [m for m in mons if not m.active and not m.fainted]
    hazards = {}
    for cond, n in conditions.items():
        key = HAZARD_MAP.get(_key(cond))
        if key:
            hazards[key] = max(hazards.get(key, 0), n if isinstance(n, int) else 1)
    return {
        "active": act,
        "fainted": fainted,
        # random doubles teams are always 6; fainted mons are revealed by the
        # time they faint, so the counter is reliable for the foe side too
        "remaining": max(0, 6 - fainted),
        "bench_known": len(bench),
        "hazards": hazards,
        "screens": {},  # folded into hazards map for features; kept for shape
    }


def battle_to_state_d(battle: DoubleBattle) -> dict:
    """State from the agent player's perspective (matches parser_doubles rows)."""
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
        "weather": weather,
        "terrain": terrain,
        "trick_room": trick_room,
        "my": _side_public_d(battle.team, battle.active_pokemon,
                             battle.side_conditions, battle.player_role),
        "foe": _side_public_d(battle.opponent_team, battle.opponent_active_pokemon,
                              battle.opponent_side_conditions, battle.opponent_role),
    }
