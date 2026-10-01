"""gauntlet.py — the current brain plays its past selves. Every game is dumped
for the learning loop, so the population feeds the corpus it competes in.

    uv run python scripts/gauntlet.py --games 60
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from poke_env import AccountConfiguration, ServerConfiguration  # noqa: E402
import poke_env.ps_client.ps_client as psc  # noqa: E402

from sb.agent import VsAgent  # noqa: E402
from sb.learn import dump_battle, learn_new_games  # noqa: E402

LOCAL = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/~~localhost/action.php?",
)

BRAINS = {
    # the morning build: full-fidelity state + everything learned so far
    "current": {"model": "v2_model.txt", "featurizer": "v2", "pi2": True},
    # dim-287 ancestor (AUC 0.7134 era) from the first commit, no pi prior
    "classic": {"model": "history/v1_classic.txt", "featurizer": "v1-classic", "pi2": False},
}


async def local_login(self, split_message):
    if self.account_configuration.password:
        r = requests.get(
            self.server_configuration.authentication_url,
            params={"act": "login", "name": self.account_configuration.username,
                    "pass": self.account_configuration.password,
                    "challstr": split_message[2] + "%7C" + split_message[3]},
            timeout=10.0)
        assertion = json.loads(r.text[1:])["assertion"]
    else:
        assertion = ""
    await self.send_message(f"/trn {self.username},0,{assertion}")
    await self.change_avatar(self._avatar)


psc.PSClient.log_in = local_login


def acc(name: str) -> AccountConfiguration:
    pw = os.environ.get(f"SHOWDOWN_{name.upper()}_PASS", "testpass123")
    return AccountConfiguration(name, pw)


def make(name: str, brain: str, tag: str) -> VsAgent:
    return VsAgent(account_configuration=acc(name), server_configuration=LOCAL,
                   battle_format="gen9randombattle", log_path=str(ROOT / "data" / f"gauntlet_{tag}.log"),
                   brain=BRAINS[brain])


async def play_game(a: VsAgent, b: VsAgent) -> str:
    await asyncio.gather(a.ps_client.logged_in.wait(), b.ps_client.logged_in.wait())
    await asyncio.gather(
        a.send_challenges(b.username, 1),
        b.accept_challenges(a.username, 1),
    )
    tag, battle = next(iter(a.battles.items()))
    if a.won:
        return "a"
    return "b"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--learn-every", type=int, default=10)
    args = ap.parse_args()

    cur = make("selfplaybot", "current", "cur")
    cls = make("classicbot", "classic", "cls")

    wins = {"current": 0, "classic": 0, "tie": 0}
    for i in range(1, args.games + 1):
        # alternate who challenges so side assignment doesn't bias
        if i % 2:
            a, b = cur, cls
        else:
            a, b = cls, cur
        try:
            # accept_challenges waits for login on the client's own loop —
            # do NOT await logged_in from this loop (cross-loop deadlock)
            await asyncio.gather(a.send_challenges(b.username, 1),
                                 b.accept_challenges(a.username, 1))
        except Exception as e:
            print(f"game {i}: setup failed: {e!r}", flush=True)
            await asyncio.sleep(3)
            continue
        done = []
        for battle in list(a.battles.values()) + list(b.battles.values()):
            dump_battle(battle)
            done.append(battle)
        bc = next(bl for bl in done if bl.battle_tag in cur.battles)
        won = cur.battles[bc.battle_tag].won
        winner = "current" if won else "classic" if cur.battles[bc.battle_tag].lost else "tie"
        wins[winner] += 1
        print(f"game {i}/{args.games}: {winner}  ({wins['current']}-{wins['classic']}-{wins['tie']})",
              flush=True)
        if i % args.learn_every == 0:
            try:
                n = await asyncio.get_event_loop().run_in_executor(None, learn_new_games)
                print(f"  learned from {n} new games", flush=True)
            except Exception as e:
                print(f"  learn failed (non-fatal): {e!r}", flush=True)
        await asyncio.sleep(2)
    n = 0
    try:
        n = await asyncio.get_event_loop().run_in_executor(None, learn_new_games)
    except Exception as e:
        print(f"final learn failed: {e!r}", flush=True)
    print(f"final learn: {n} games; score {wins}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
