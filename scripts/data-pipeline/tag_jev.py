#!/usr/bin/env python3
"""tag_jev.py — semantic tagging pass over the needsSemantic-flagged entities.

reads   game_data.json + tags.json (next to this script)
writes  tags_semantic.json   (id -> judgment result, resumable)

usage:
  TYPESAFE_API_KEY=... python3 tag_jev.py --limit 8        # pilot
  TYPESAFE_API_KEY=... python3 tag_jev.py                  # full flagged set
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.typesafe.ai/v1/systemone"

ROLE_CRITERIA = {
    "offense": "Boosts damage dealt, attack stats, crit, or enables KOs",
    "defense": "Reduces damage taken, heals, or prevents hits",
    "speed": "Acts earlier: priority, speed boosts, paralysis, trick room",
    "momentum": "Controls switching: pivoting, entry effects, trapping, forced switches",
    "field": "Sets or modifies weather, terrain, hazards, screens",
    "utility": "None of the above dominates (consumables, scouting, niche)",
}
IMPACT_LEVELS = [
    "Detrimental or useless",
    "Niche — rarely decides anything",
    "Relevant — appears in real games",
    "Meta-relevant — warps team choices",
]
NOULS = [
    ("modifies_damage", "Does it change the amount of damage a Pokemon deals or takes?"),
    ("affects_switching", "Does it cause, enable, prevent, or punish switching (including on entry or on contact)?"),
    ("triggers_on_entry", "Does it do something when a Pokemon enters the field?"),
]
TAG_T = 0.2   # role probability above this -> tag
NOUL_T = 0.5  # noul above this -> tag
CONF_T = 0.4  # role confidence below this -> keep needsSemantic (human review)


def build_request(batch):
    """batch: list of entity dicts -> (request_body, id_map). rubrics hoisted into
    state once — per-question instructions just point at them (cheaper per entity)."""
    entities, questions, id_map = [], {}, {}
    for i, e in enumerate(batch):
        p = f"entities[{i}]"
        entities.append({
            "kind": e["kind"], "id": e["id"], "name": e["name"],
            "effect": e["effect"], "detail": e.get("detail", ""),
            "structured": e.get("structured", {}),
        })
        id_map[f"e{i}"] = (e["kind"], e["id"])
        questions[f"e{i}_role"] = {
            "type": "choice",
            "instructions": (f"For the Pokemon {e['kind']} `{p}.name` with effect `{p}.effect`: "
                             "which option in `rubrics.role` is its PRIMARY battle role? "
                             "Judge by what it does most of the time it matters."),
            "criteria": ROLE_CRITERIA,
        }
        for slot, qtext in NOULS:
            questions[f"e{i}_{slot}"] = {
                "type": "noul",
                "instructions": f"For `{p}.name` (`{p}.effect`): {qtext}",
            }
        questions[f"e{i}_impact"] = {
            "type": "score",
            "instructions": (f"How strong is `{p}.name` in competitive singles (gen 9)? "
                             "Rate against the levels in `rubrics.impact`."),
            "criteria": IMPACT_LEVELS,
        }
    state = {"rubrics": {"role": ROLE_CRITERIA, "impact": IMPACT_LEVELS}, "entities": entities}
    return {"model": "jev-latest", "state": state, "questions": questions}, id_map


def call_api(body, key, max_retries=5):
    data = json.dumps(body).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(API, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def fold(answers, prefix):
    """one entity's answers -> tags.json-shaped entry"""
    role = answers[f"{prefix}_role"]
    tags = {opt for opt, p in role["probabilities"].items() if p > TAG_T}
    tags.discard("utility") if len(tags) > 1 else None  # utility only kept if it's the sole mass
    for slot, _ in NOULS:
        if answers[f"{prefix}_{slot}"]["noul"] > NOUL_T:
            tags.add(slot)
    impact = answers[f"{prefix}_impact"]
    return {
        "tags": sorted(tags),
        "scores": {"impact": round(impact["score"] / (len(IMPACT_LEVELS) - 1), 3)},
        "confidence": "semantic",
        "needsSemantic": role["confidence"] < CONF_T,
        "jev": {
            "roleProbabilities": {k: round(v, 3) for k, v in role["probabilities"].items()},
            "roleConfidence": round(role["confidence"], 3),
            "impactProbabilities": {k: round(v, 3) for k, v in impact["probabilities"].items()},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("set TYPESAFE_API_KEY")

    gd = json.load(open(f"{HERE}/game_data.json"))
    tags = json.load(open(f"{HERE}/tags.json"))
    out_path = f"{HERE}/tags_semantic.json"
    done = json.load(open(out_path)) if os.path.exists(out_path) else {}

    pending = []
    for kind, table, src in (("ability", gd["abilities"], tags["abilities"]),
                             ("item", gd["items"], tags["items"]),
                             ("move", gd["moves"], tags["moves"])):
        for eid, t in src.items():
            if not t.get("needsSemantic") or f"{kind}:{eid}" in done:
                continue
            e = table[eid]
            pending.append({"kind": kind, "id": eid, "name": e["name"],
                            "effect": e.get("shortDesc") or e.get("desc") or "",
                            "detail": e.get("desc", ""),
                            "structured": {k: e[k] for k in ("rating", "hasScripts", "isNonstandard") if k in e}})
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(pending)} entities to judge, batch={args.batch}, "
          f"{(len(pending) + args.batch - 1) // args.batch} calls", file=sys.stderr)

    tot_in = tot_out = 0
    for off in range(0, len(pending), args.batch):
        batch = pending[off: off + args.batch]
        body, id_map = build_request(batch)
        resp = call_api(body, key)
        usage = resp.get("usage", {})
        tot_in += usage.get("input_tokens", 0)
        tot_out += usage.get("output_tokens", 0)
        for qid, (kind, eid) in id_map.items():
            entry = fold(resp["answers"], qid)
            if kind == "item":
                e = gd["items"][eid]
                if e.get("isBerry") or "single use" in (e.get("shortDesc") or "").lower():
                    entry["tags"] = sorted(set(entry["tags"]) | {"consumable"})
            done[f"{kind}:{eid}"] = entry
        json.dump(done, open(out_path, "w"), indent=1)
        print(f"  [{off + len(batch)}/{len(pending)}] tokens so far: in={tot_in} out={tot_out}", file=sys.stderr)
        if off + args.batch < len(pending):
            time.sleep(args.sleep)

    print(json.dumps({"judged": len(done), "input_tokens": tot_in, "output_tokens": tot_out}))


if __name__ == "__main__":
    main()
