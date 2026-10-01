"""ladder.py — kingambit plays rated randbats on the MAIN showdown server.

Usage: uv run python scripts/ladder.py --games 3

GOTCHA: poke-env's Player defaults to LocalhostServerConfiguration — always
pass ShowdownServerConfiguration explicitly here, or the bot cheerfully
logs into localhost and main assertions die with 'verification signature
was invalid' (local Verifier uses our RSA-SHA256 key, main uses RSA-SHA1).

Stock poke-env login works on main (the %7C in its challstr is fine).
Every finished game is dumped into the learning pool, same as local games:
main-server replays fold into rows_personal on the next learn cycle.
Announces each battle's public spectate URL the moment it starts.
"""
import argparse
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from poke_env import AccountConfiguration, ShowdownServerConfiguration  # noqa: E402

from sb.agent import VsAgent  # noqa: E402
from sb.learn import dump_battle  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    args = ap.parse_args()

    creds = json.load(open(ROOT / "data" / "ladder-account.json"))
    agent = VsAgent(
        account_configuration=AccountConfiguration(creds["username"], creds["password"]),
        server_configuration=ShowdownServerConfiguration,
        battle_format="gen9randombattle",
        log_path=str(ROOT / "data" / "ladder-games.log"),
        brain={"model": "v2_model.txt", "featurizer": "v2", "pi2": True},
    )

    seen = set()

    async def announce() -> None:
        while True:
            for tag in agent.battles:
                if tag not in seen:
                    seen.add(tag)
                    room = tag if tag.startswith("battle-") else f"battle-{tag}"
                    print(f"BATTLE {room} https://play.pokemonshowdown.com/{room}", flush=True)
                    # no more infinite stalls: main has no auto-timer in randbats
                    await agent.ps_client.send_message("/timer on", room=room)
            await asyncio.sleep(2)

    announcer = asyncio.create_task(announce())
    print(f"laddering as {creds['username']} for {args.games} games...", flush=True)
    try:
        await agent.ladder(args.games)
    finally:
        announcer.cancel()

    for tag, b in agent.battles.items():
        dump_battle(b)
        result = "WIN" if b.won else "LOSS" if b.lost else "TIE"
        print(f"{tag}: {result} (turns: {b.turn})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
