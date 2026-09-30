"""harvest.py — download replay logs from replay.pokemonshowdown.com."""
import argparse
import json
import pathlib
import time
import urllib.request

BASE = "https://replay.pokemonshowdown.com"
UA = {"User-Agent": "showdown-brain/0.1 (replay research)"}

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(format_, page):
    return json.loads(get(f"{BASE}/search.json?format={format_}&page={page}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--rating-min", type=int, default=None,
                    help="only games where both players >= rating")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = (RAW / "manifest.jsonl").open("a")
    seen = {p.stem for p in RAW.glob("*.log")}
    got, page = 0, 1
    while got < args.limit:
        try:
            batch = search(args.format, page)
        except Exception as e:
            print(f"search page {page} failed: {e}; backing off")
            time.sleep(5)
            continue
        if not batch:
            print("no more results")
            break
        for meta in batch:
            if got >= args.limit:
                break
            rid = meta["id"]
            if rid in seen or meta.get("private"):
                continue
            if args.rating_min and not (meta.get("rating") or 0) >= args.rating_min:
                continue
            try:
                log = get(f"{BASE}/{rid}.log")
            except Exception as e:
                print(f"fetch {rid} failed: {e}")
                continue
            (RAW / f"{rid}.log").write_bytes(log)
            manifest.write(json.dumps({"id": rid, "rating": meta.get("rating"),
                                       "players": meta.get("players")}) + "\n")
            manifest.flush()
            seen.add(rid)
            got += 1
            time.sleep(args.sleep)
        print(f"page {page}: {got}/{args.limit} collected", flush=True)
        page += 1
        time.sleep(args.sleep)
    print(f"done: {got} logs in {RAW}")


if __name__ == "__main__":
    main()
