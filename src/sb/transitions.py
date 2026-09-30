"""transitions.py — mainline physics for one-ply lookahead, in dict-space.

Damage formula (gen 9, expected roll): handles stats, boosts, STAB, type
matrix, weather/terrain, burn, accuracy. Unmodelable move effects are the
caller's problem (agent falls back to prior/default).
"""
import copy
import json

from sb.features import DATA as _FDATA, resolve_species, to_id

DATA = _FDATA

MOVES = None
MATRIX = None
SPECIES = None
SETS = None


def load():
    global MOVES, MATRIX, SPECIES, SETS
    gd = json.load(open(DATA / "game_data.json"))
    MOVES = gd["moves"]
    MATRIX = gd["matrix"]
    SPECIES = gd["species"]
    SETS = json.load(open(DATA / "randbats-sets.json"))


def _ensure():
    """Self-load for consumers that never call load() (e.g. feature builders
    that only reach us through _momentum)."""
    if SPECIES is None:
        load()


BOOSTS = ["atk", "def", "spa", "spd", "spe"]


def effective_speed(mon: dict, state: dict) -> float:
    _ensure()
    """Speed for ordering comparisons. Trick Room inverts (returned as a
    negative so 'higher = acts first' still holds). Paralysis halves."""
    if not mon:
        return 0.0
    sp = estimate_stats(resolve_species(mon.get("species", "")),
                        mon.get("level", 80))["spe"]
    sp *= boost_mult(mon.get("boosts", {}).get("spe", 0))
    if mon.get("status") == "par":
        sp *= 0.5
    return -sp if state.get("trick_room") else sp


def damaging_moves(mon: dict, pool_from_sets: bool = True) -> list[str]:
    _ensure()
    """Known damaging moves, or the species' randbats movepool as a prior."""
    moves = [m for m in mon.get("moves", [])
             if MOVES.get(m, {}).get("power")
             and MOVES.get(m, {}).get("category") != "status"]
    if moves or not pool_from_sets:
        return moves
    entry = SETS.get(resolve_species(mon.get("species", "")))
    if not entry:
        return []
    s = set()
    for st in entry["sets"]:
        s.update(to_id(m) for m in st["movepool"])
    return [m for m in s
            if MOVES.get(m, {}).get("power")
            and MOVES.get(m, {}).get("category") != "status"]


def kill_clock(attacker: dict, defender: dict, state: dict) -> int:
    _ensure()
    """Turns of best-move damage for `attacker` to KO `defender` (1, 2, or 3
    meaning 'three or more' = not pressuring). The who-forces-who clock."""
    if not attacker or not defender or defender["hp"] <= 0:
        return 1
    moves = damaging_moves(attacker)
    if not moves:
        return 3
    best = max(damage(attacker, defender, m, state) for m in moves)
    if best >= defender["hp"]:
        return 1
    if best * 2 >= defender["hp"]:
        return 2
    return 3


def has_priority(mon: dict) -> bool:
    _ensure()
    """Does this mon run a damaging priority move (known or from its sets)?"""
    return any((MOVES.get(m, {}).get("priority") or 0) > 0
               for m in damaging_moves(mon))


def boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage > 0 else 2 / (2 - stage)


def estimate_stats(species_id: str, level: int, evs: dict | None = None) -> dict:
    _ensure()
    """HP + five stats; EVs default to randbats-ish 85 flat unless given."""
    sp = SPECIES.get(resolve_species(species_id), {})
    bs = sp["baseStats"]
    evs = evs or {}
    out = {}
    for st in ["hp", "atk", "def", "spa", "spd", "spe"]:
        ev = evs.get(st, 85)
        if st == "hp":
            out[st] = int((2 * bs[st] + 31 + ev // 4) * level / 100) + level + 10
        else:
            out[st] = int((2 * bs[st] + 31 + ev // 4) * level / 100) + 5
    return out


def avg_set_evs(species_id: str) -> dict:
    _ensure()
    """Average EV spread across that species' randbats sets (our belief prior)."""
    entry = SETS.get(resolve_species(species_id))
    if not entry or not entry.get("sets"):
        return {}
    keys = ["hp", "atk", "def", "spa", "spd", "spe"]
    acc = {k: 0 for k in keys}
    n = 0
    for s in entry["sets"]:
        evs = s.get("evs") or {}
        for k in keys:
            acc[k] += evs.get(k, 0)
        n += 1
    return {k: acc[k] // n for k in keys} if n else {}


def tera_prior(species_id: str) -> str | None:
    _ensure()
    """First listed randbats tera type for a species (our pre-tera belief)."""
    entry = SETS.get(resolve_species(species_id))
    if entry and entry.get("sets"):
        ts = entry["sets"][0].get("teraTypes") or []
        if ts:
            return ts[0]
    return None


def defender_types(mon: dict) -> list[str]:
    _ensure()
    if mon.get("tera"):
        return [mon["tera"]]
    sp = SPECIES.get(resolve_species(mon["species"]), {})
    if "types" not in sp and sp.get("baseSpecies"):
        sp = SPECIES.get(sp["baseSpecies"], {})
    return sp.get("types", [])


def effectiveness(move_type: str, mon: dict) -> float:
    _ensure()
    mult = 1.0
    for t in defender_types(mon):
        mult *= MATRIX.get(move_type, {}).get(t, 1)
    return mult


def weather_mult(move_type: str, weather) -> float:
    if weather == "raindance":
        return 1.5 if move_type == "Water" else 0.5 if move_type == "Fire" else 1.0
    if weather == "sunnyday":
        return 1.5 if move_type == "Fire" else 0.5 if move_type == "Water" else 1.0
    return 1.0


def terrain_mult(move_type: str, terrain, grounded: bool = True) -> float:
    if not grounded:
        return 1.0
    return {"electricterrain": {"Electric": 1.5}, "grassyterrain": {"Grass": 1.5},
            "psychicterrain": {"Psychic": 1.5}}.get(terrain, {}).get(move_type, 1.0)


def damage(attacker: dict, defender: dict, move_id: str, state: dict,
           att_stats: dict | None = None, def_stats: dict | None = None) -> float:
    """Expected damage fraction of defender's max HP (0 if status/no power)."""
    _ensure()
    entry = MOVES.get(move_id)
    if not entry or entry["category"] == "status" or not entry.get("power"):
        return 0.0
    if effectiveness(entry["type"], defender) == 0:
        return 0.0
    att_stats = att_stats or estimate_stats(attacker["species"], attacker.get("level", 80))
    def_stats = def_stats or estimate_stats(defender["species"], defender.get("level", 80))
    ab = BOOSTS
    cat = entry["category"]
    a_stat = "atk" if cat == "physical" else "spa"
    d_stat = "def" if cat == "physical" else "spd"
    a = att_stats[a_stat] * boost_mult(attacker.get("boosts", {}).get(a_stat, 0))
    d = def_stats[d_stat] * boost_mult(defender.get("boosts", {}).get(d_stat, 0))
    if cat == "physical" and attacker.get("status") == "brn":
        a *= 0.5
    level = attacker.get("level", 80)
    base = ((2 * level / 5 + 2) * entry["power"] * a / max(1, d)) / 50 + 2
    stab = 1.5 if entry["type"] in defender_types(attacker) or entry["type"] == attacker.get("tera") else 1.0
    eff = effectiveness(entry["type"], defender)
    mult = (stab * eff * weather_mult(entry["type"], state.get("weather"))
            * terrain_mult(entry["type"], state.get("terrain"))
            * 0.925)
    acc = entry.get("accuracy")
    if acc:
        mult *= acc / 100
    dmg = base * mult
    return min(1.5, dmg / max(1, def_stats["hp"]))


def apply_boosts(mon: dict, boosts: dict, sign: int = 1):
    for st, n in (boosts or {}).items():
        if st in BOOSTS or st in ("accuracy", "evasion"):
            cur = mon.setdefault("boosts", {}).get(st, 0)
            mon["boosts"][st] = max(-6, min(6, cur + sign * n))


def apply_move_effects(state: dict, actor_side: str, move_id: str) -> bool:
    """Apply mainline non-damage effects of a move to a state copy.
    Returns True if handled; False => caller's fallback (prior)."""
    entry = MOVES.get(move_id)
    if not entry:
        return False
    me = state[actor_side]["active"]
    them = state["foe" if actor_side == "my" else "my"]["active"]
    handled = False
    if entry.get("boosts") and me:  # self-boosting status (Swords Dance etc.)
        apply_boosts(me, entry["boosts"])
        handled = True
    self_eff = entry.get("self") or {}
    if isinstance(self_eff, dict) and self_eff.get("boosts") and me:
        apply_boosts(me, self_eff["boosts"])
        handled = True
    if entry.get("heal") and me:
        me["hp"] = min(1.0, me["hp"] + entry["heal"][0] / entry["heal"][1])
        handled = True
    if entry.get("status") and them:
        if not them.get("status"):
            them["status"] = entry["status"]
        handled = True
    if entry.get("sideCondition"):
        side = state["foe" if actor_side == "my" else "my"]
        cond = entry["sideCondition"]
        side["hazards"][cond] = side["hazards"].get(cond, 0) + 1
        handled = True
    if entry.get("weather"):
        state["weather"] = {"raindance": "raindance", "sunnyday": "sunnyday",
                            "sandstorm": "sandstorm", "snowscape": "snow"}.get(entry["weather"])
        handled = True
    if entry.get("terrain"):
        state["terrain"] = entry["terrain"]
        handled = True
    # damage-with-secondary (boost drops on target etc.) ignored in v0
    return handled


def apply_entry_hazards(mon: dict, hazards: dict):
    """Charge a just-switched-in mon its entry hazards (mutates in place):
    stealth rock (type-aware, up to 1/2 vs 4x weak), spikes per layer
    (grounded only), toxic spikes (status), sticky web (-1 spe)."""
    _ensure()
    if not mon or mon["hp"] <= 0:
        return
    hp_loss = 0.0
    if hazards.get("stealthrock"):
        hp_loss += effectiveness("Rock", mon) / 8
    if hazards.get("spikes"):
        grounded = "Flying" not in defender_types(mon) \
            and mon.get("ability") != "levitate"
        if grounded:
            hp_loss += min(3, hazards["spikes"]) / 8
    if hp_loss:
        mon["hp"] = max(0.0, mon["hp"] - hp_loss)
    if hazards.get("toxicspikes") and not mon.get("status"):
        if not ({"Poison", "Steel"} & set(defender_types(mon))):
            mon["status"] = "tox" if hazards["toxicspikes"] >= 2 else "psn"
    if hazards.get("stickyweb"):
        apply_boosts(mon, {"spe": -1})


def clone(state: dict) -> dict:
    return copy.deepcopy(state)
