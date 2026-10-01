"""build_bundle.py — pack all game data into ONE compressible JSON document.

Design rules (smallest self-describing JSON):
  - every entity list is `{"ids": [...canonical ids...], "rows": [[...ints...]]}`
    — strings appear exactly once, everything else references by INDEX;
  - matrices are FLAT row-major int arrays (typechart encoded x4: 0,1,2,4,8,16);
  - entity rows are fixed-width int arrays with the field order in _schema;
  - tags are sorted index lists over a small tag vocabulary;
  - no prose fields (descriptions live in the upstream pokedex).

Reads ~/pokemon-data (or $POKEMON_DATA). Writes assets/gamebundle.json.
Round-trip helpers: load_bundle() + index lookups, validated on build.
"""
import gzip
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("POKEMON_DATA", pathlib.Path.home() / "pokemon-data"))
OUT = ROOT / "assets" / "gamebundle.json"

TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
         "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
         "Steel", "Fairy", "Stellar"]
CATEGORIES = ["Physical", "Special", "Status"]


def main():
    gd = json.load(open(DATA / "game_data.json"))
    tags = json.load(open(DATA / "tags_final.json"))
    sets_s = json.load(open(DATA / "randbats-sets.json"))
    sets_d = json.load(open(DATA / "randdoubles-sets.json"))

    # ---- vocabularies -------------------------------------------------------
    move_ids = sorted(gd["moves"])
    ability_ids = sorted(gd["abilities"])
    item_ids = sorted(gd["items"])
    species_ids = sorted(s for s, v in gd["species"].items() if "baseStats" in v)
    targets = sorted({m.get("target") for m in gd["moves"].values() if m.get("target")})
    tag_vocab = {
        "item": sorted({t for e in tags["items"].values() for t in e.get("tags", [])}),
        "ability": sorted({t for e in tags["abilities"].values() for t in e.get("tags", [])}),
    }
    role_vocab = sorted({s["role"] for table in (sets_s, sets_d)
                         for e in table.values() for s in e["sets"] if s.get("role")})

    mv = {t: i for i, t in enumerate(move_ids)}
    av = {a: i for i, a in enumerate(ability_ids)}
    iv = {i: i for i, i in enumerate(item_ids)}
    sv = {s: i for i, s in enumerate(species_ids)}
    tv = {t: i for i, t in enumerate(targets)}
    rv = {r: i for i, r in enumerate(role_vocab)}
    itv = {t: i for i, t in enumerate(tag_vocab["item"])}
    atv = {t: i for i, t in enumerate(tag_vocab["ability"])}

    def ti(t):  # type index, -1 pad
        return TYPES.index(t) if t in TYPES else -1

    # ---- typechart: flat row-major x4 ints ---------------------------------
    matrix = gd["matrix"]
    chart = []
    for at in TYPES:
        row = matrix.get(at, {})
        for dt in TYPES:
            x = row.get(dt, 1)
            chart.append(int(x * 4))  # 0,1,2,4,8,16

    # ---- entity rows --------------------------------------------------------
    move_rows = []
    for m in move_ids:
        e = gd["moves"][m]
        move_rows.append([ti(e["type"]), CATEGORIES.index(e["category"].title()),
                          e.get("power") or 0, e.get("accuracy") or 0,
                          e.get("pp") or 0, e.get("priority") or 0,
                          tv.get(e.get("target"), -1)])

    def spec_row(s):
        e = gd["species"][s]
        t = [ti(x) for x in e["types"]] + [-1] * (2 - len(e["types"]))
        ab = e.get("abilities", {})
        a_idx = [av[to_id(ab[k])] for k in ("0", "1", "H") if to_id(ab.get(k, "")) in av]
        a_idx += [-1] * (3 - len(a_idx))
        bs = e["baseStats"]
        return t + [bs["hp"], bs["atk"], bs["def"], bs["spa"], bs["spd"], bs["spe"]] + a_idx

    species_rows = [spec_row(s) for s in species_ids]

    item_tags = [sorted(itv[t] for t in tags["items"].get(i, {}).get("tags", []))
                 for i in item_ids]
    abil_tags = [sorted(atv[t] for t in tags["abilities"].get(a, {}).get("tags", []))
                 for a in ability_ids]

    def pack_sets(table):
        out = {}
        for s, e in table.items():
            if s not in sv:
                continue
            sets = []
            for st in e["sets"]:
                moves = sorted({mv[to_id(m)] for m in st["movepool"] if to_id(m) in mv})
                abils = sorted({av[to_id(a)] for a in st.get("abilities", []) if to_id(a) in av})
                teras = sorted({ti(x) for x in st.get("teraTypes", [])})
                evs = st.get("evs") or {}
                ev_row = [evs.get(k, 0) for k in ("hp", "atk", "def", "spa", "spd", "spe")]
                sets.append([rv.get(st.get("role"), -1), moves, abils, teras, ev_row])
            out[str(sv[s])] = {"level": e.get("level", 80), "sets": sets}
        return out

    bundle = {
        "_schema": ("gamebundle v1. all entity tables are {ids, rows/tags/sets}; "
                    "cross-references are INTEGER INDICES into the named vocab. "
                    "typechart: flat row-major over types[], damage multiplier x4 "
                    "(0=immune,1=0.25x,2=0.5x,4=1x,8=2x,16=4x). "
                    "move rows: [type, category, power, accuracy, pp, priority, target]. "
                    "species rows: [type1, type2(-1 pad), hp, atk, def, spa, spd, spe, "
                    "ability0, ability1, abilityH(-1 pad)]. "
                    "set rows: [role, [moveIdx], [abilityIdx], [teraTypeIdx], [evs x6]]."),
        "vocab": {"types": TYPES, "categories": CATEGORIES, "targets": targets,
                  "tags": tag_vocab, "roles": role_vocab},
        "typechart": chart,
        "moves": {"ids": move_ids, "rows": move_rows},
        "abilities": {"ids": ability_ids, "tags": abil_tags},
        "items": {"ids": item_ids, "tags": item_tags},
        "species": {"ids": species_ids, "rows": species_rows},
        "randbats": {"singles": pack_sets(sets_s), "doubles": pack_sets(sets_d)},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(bundle, separators=(",", ":"))
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(raw)  # atomic publish: temp + rename so concurrent readers
    tmp.replace(OUT)     # never see a torn file (one got interleaved once)
    gz = gzip.compress(raw.encode(), compresslevel=9)
    (OUT.parent / "gamebundle.json.gz").write_bytes(gz)

    # ---- round-trip validation ----------------------------------------------
    b = json.loads(raw)
    assert b["typechart"][TYPES.index("Dragon") * 19 + TYPES.index("Fairy")] == 0
    qm = b["moves"]["ids"].index("earthquake")
    assert b["moves"]["rows"][qm][2] == 100
    qs = b["species"]["ids"].index("garchomp")
    assert b["species"]["rows"][qs][2] == 108  # atk slot
    sp = b["randbats"]["singles"][str(qs)]
    assert sp["sets"]
    print(f"bundle OK: {len(raw)/1e6:.2f} MB raw, {len(gz)/1e3:.0f} KB gz")
    for name, table in (("moves", b["moves"]), ("abilities", b["abilities"]),
                        ("items", b["items"]), ("species", b["species"])):
        print(f"  {name}: {len(table['ids'])}")
    print(f"  sets: {sum(len(v['sets']) for v in b['randbats']['singles'].values())} singles / "
          f"{sum(len(v['sets']) for v in b['randbats']['doubles'].values())} doubles")


def to_id(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def load_bundle(path=None):
    return json.load(open(path or OUT))


if __name__ == "__main__":
    sys.exit(main())
