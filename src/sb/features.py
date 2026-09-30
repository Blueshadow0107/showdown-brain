"""features.py — state dict -> fixed-length numeric vector.

Reads species stats and semantic tags from the repo's `assets/` dir (a copy
of the pokemon-data game files). Override with the POKEMON_DATA env var.
The vector is perspective-symmetric: built identically for `my` and `foe`,
swapped by caller.
"""
import json
import os
import pathlib
import re

_ASSETS = pathlib.Path(__file__).resolve().parent.parent.parent / "assets"
DATA = pathlib.Path(os.environ.get(
    "POKEMON_DATA",
    _ASSETS if _ASSETS.exists() else pathlib.Path.home() / "pokemon-data"))


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
         "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
         "Steel", "Fairy", "Stellar"]
STATUSES = ["brn", "par", "psn", "tox", "slp", "frz"]
BOOSTS = ["atk", "def", "spa", "spd", "spe"]
WEATHERS = ["rain", "sun", "sand", "snow", "other"]
TERRAINS = ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain", "other"]
VOLATILES = [  # top conditions by competitive relevance
    "protect", "substitute", "taunt", "encore", "disable", "leechseed", "confusion",
    "trap", "partiallytrapped", "attract", "curse", "embargo", "torment", "nightmare",
    "focusenergy", "ingrain", "aquaring", "magnetrise", "slowstart", "unburden",
]
ITEM_TAGS = ["choice", "consumable", "offense", "defense", "speed", "recovery",
             "hazardinteract", "switching", "weather", "terrain", "berry"]
ABILITY_TAGS = ["offense", "defense", "speed", "momentum", "field", "immunity",
                "recovery", "switching", "priority", "boost", "entry", "contact",
                "weather", "terrain", "consumable", "modifiesdamage"]

WEATHER_MAP = {"raindance": "rain", "primordialsea": "rain", "heavyrain": "rain",
               "sunnyday": "sun", "desolateland": "sun", "harshsun": "sun",
               "sandstorm": "sand", "snow": "snow", "hail": "snow"}


def _load():
    global SPECIES, ITEM_TAGMAP, ABIL_TAGMAP, FEATURE_NAMES
    SPECIES = json.load(open(DATA / "species.json"))
    tags = json.load(open(DATA / "tags_final.json"))
    ITEM_TAGMAP = {eid: set(t.get("tags", [])) for eid, t in tags["items"].items()}
    ABIL_TAGMAP = {eid: set(t.get("tags", [])) for eid, t in tags["abilities"].items()}
    FEATURE_NAMES = None


_FORMES = None


def _forme_map() -> dict:
    """forme-id -> base-species-id, from poke-env's dex (which carries every
    cosmetic forme). Lazily built; empty if poke-env data is unavailable."""
    global _FORMES
    if _FORMES is None:
        try:
            from poke_env.data import GenData
            dex = GenData(9).pokedex
            _FORMES = {k: to_id(v["baseSpecies"]) for k, v in dex.items()
                       if isinstance(v, dict) and v.get("baseSpecies")}
        except Exception:
            _FORMES = {}
    return _FORMES


def resolve_species(species_id: str) -> str:
    """Follow baseSpecies chains so cosmetic/inherited formes (florgesblue,
    burmysandy, deerlingwinter, ...) map to the entry that actually carries
    baseStats. Absent ids are returned unchanged (lookups then default)."""
    sid = species_id
    for _ in range(3):
        sp = SPECIES.get(sid) if SPECIES else None
        if sp is not None:
            if "baseStats" in sp or not sp.get("baseSpecies"):
                return sid
            sid = to_id(sp["baseSpecies"])
            continue
        base = _forme_map().get(sid)  # cosmetic forme missing from species.json
        if not base:
            return sid
        sid = base
    return sid


def one_hot(idx, n):
    return [1.0 if i == idx else 0.0 for i in range(n)]


def mon_features(m: dict | None) -> list[float]:
    if not m:
        return [0.0] * MON_DIM
    v = []
    sp = SPECIES.get(resolve_species(m["species"]))
    if sp and "types" not in sp and sp.get("baseSpecies"):
        sp = SPECIES.get(to_id(sp["baseSpecies"]), sp)  # cosmetic formes inherit base stats
    if sp and "types" in sp:
        t1 = TYPES.index(sp["types"][0]) if sp["types"][0] in TYPES else len(TYPES) - 1
        v += one_hot(t1, len(TYPES))
        if len(sp["types"]) > 1 and sp["types"][1] in TYPES:
            v += one_hot(TYPES.index(sp["types"][1]), len(TYPES))
        else:
            v += [0.0] * len(TYPES)
        bs = sp["baseStats"]
        v += [bs["hp"] / 255, bs["atk"] / 190, bs["def"] / 230, bs["spa"] / 194,
              bs["spd"] / 230, bs["spe"] / 180]
        v += [1.0]  # species known
    else:
        v += [0.0] * (2 * len(TYPES) + 6) + [0.0]
    v += [m["hp"], 1.0 if m["hp"] == 0 else 0.0]
    st = m["status"]
    v += one_hot(STATUSES.index(st), len(STATUSES)) if st in STATUSES else [0.0] * len(STATUSES)
    v += [m["boosts"].get(b, 0) / 6 for b in BOOSTS]
    vol = set(m["volatiles"])
    v += [1.0 if x in vol else 0.0 for x in VOLATILES]
    it = m["item"]
    it_tags = ITEM_TAGMAP.get(it, set()) if it else set()
    v += [1.0 if it else 0.0] + [1.0 if t in it_tags else 0.0 for t in ITEM_TAGS]
    ab = m["ability"]
    ab_tags = ABIL_TAGMAP.get(ab, set()) if ab else set()
    v += [1.0 if ab else 0.0] + [1.0 if t in ab_tags else 0.0 for t in ABILITY_TAGS]
    tera = m["tera"]
    v += one_hot(TYPES.index(tera), len(TYPES)) if tera in TYPES else [0.0] * len(TYPES)
    v += [min(len(m["moves"]), 4) / 4]
    return v


MON_DIM = len(one_hot(0, len(TYPES))) * 2 + 6 + 1 + 2 + len(STATUSES) + len(BOOSTS) \
    + len(VOLATILES) + 1 + len(ITEM_TAGS) + 1 + len(ABILITY_TAGS) + len(TYPES) + 1


def side_features(s: dict) -> list[float]:
    hz = s["hazards"]
    sc = s.get("screens", {})
    scr = lambda k: 1.0 if (sc.get(k) or hz.get(k)) else 0.0
    v = [s["fainted"] / 6, s["remaining"] / 6, s["bench_known"] / 5]
    v += [1.0 if hz.get("stealthrock") else 0.0,
          hz.get("spikes", 0) / 3, hz.get("toxicspikes", 0) / 2,
          1.0 if hz.get("stickyweb") else 0.0]
    v += [scr("reflect"), scr("lightscreen"), scr("auroraveil"), scr("tailwind")]
    v += mon_features(s["active"])
    return v


def state_features(state: dict) -> list[float]:
    w = WEATHER_MAP.get(state["weather"], "other") if state["weather"] else None
    t = state["terrain"] if state["terrain"] in TERRAINS else ("other" if state["terrain"] else None)
    v = one_hot(WEATHERS.index(w), len(WEATHERS)) if w else [0.0] * len(WEATHERS)
    v += one_hot(TERRAINS.index(t), len(TERRAINS)) if t else [0.0] * len(TERRAINS)
    v += [1.0 if state["trick_room"] else 0.0]
    v += side_features(state["my"]) + side_features(state["foe"])
    v += _momentum(state)
    return v


def _momentum(state: dict) -> list[float]:
    """Tempo/pressure block: speed control, kill-clocks (who forces who),
    priority presence, and switch counters (the cascade meter). Lazy import:
    transitions imports features at module level."""
    from sb import transitions as T
    my, foe = state["my"]["active"], state["foe"]["active"]
    v: list[float] = []
    if my and foe and my["hp"] > 0 and foe["hp"] > 0:
        ms, fs = T.effective_speed(my, state), T.effective_speed(foe, state)
        v += one_hot(0 if ms > fs * 1.05 else (2 if fs > ms * 1.05 else 1), 3)
        v += one_hot(T.kill_clock(my, foe, state) - 1, 3)
        v += one_hot(T.kill_clock(foe, my, state) - 1, 3)
        v += [1.0 if T.has_priority(my) else 0.0,
              1.0 if T.has_priority(foe) else 0.0]
    else:
        v += [0.0] * 11
    v += [min(state["my"].get("switches", 0), 6) / 6,
          min(state["foe"].get("switches", 0), 6) / 6]
    return v


MOMENTUM_DIM = 13


DIM = len(WEATHERS) + len(TERRAINS) + 1 + 2 * (3 + 4 + 4 + MON_DIM) + MOMENTUM_DIM


def build_matrix(rows):
    _load()
    import numpy as np
    X = np.zeros((len(rows), DIM), dtype="float32")
    y = np.zeros(len(rows), dtype="float32")
    g = np.empty(len(rows), dtype="U40")
    for i, r in enumerate(rows):
        X[i] = state_features(r["state"])
        y[i] = r["outcome"]
        g[i] = r["game"]
    return X, y, g
