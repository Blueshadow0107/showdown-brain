"""showdown_state_ffa.py — FFABattle -> pairwise 2-key state dict (for V(s)).

FFA projects the 4-player game into one ordinary singles-shaped state per
live opponent: `my` is our side, `foe` is the OpponentView with n == foe_n.
Same dict shapes as showdown_state.py so features_v2.state_features_v2
consumes it unchanged. switch_counts: optional {"my": n, foe_n: n} momentum
counters maintained by the agent (parser rows carry their own).
"""
from sb.ffa_poke_env import FFABattle
from sb.showdown_state import TERRAIN_MAP, WEATHER_MAP, _key, _side_public


def battle_to_state_ffa(battle: FFABattle, foe_n: int,
                        switch_counts: dict | None = None) -> dict:
    view = next((v for v in battle.opponents if v.n == foe_n), None)
    if view is None:
        raise ValueError(f"no opponent view for side p{foe_n + 1}")
    sw = switch_counts or {}
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
                           tera_used=bool(getattr(battle, "_used_tera", False))),
        "foe": _side_public(view.team, view.active, view.side_conditions,
                            sw.get(foe_n, 0), tera_used=view.tera_used),
    }
