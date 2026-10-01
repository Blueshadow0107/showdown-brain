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
from sb.agent_doubles import VsAgentDoubles  # noqa: E402
from sb.learn import dump_battle  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--challenge", default=None,
                    help="challenge this user instead of laddering")
    ap.add_argument("--accept", default=None,
                    help="wait for a challenge from this user (omit value for anyone)")
    args = ap.parse_args()

    creds = json.load(open(ROOT / "data" / "ladder-account.json"))
    kwargs = dict(
        account_configuration=AccountConfiguration(creds["username"], creds["password"]),
        server_configuration=ShowdownServerConfiguration,
        battle_format=args.format,
    )
    if "doubles" in args.format:
        agent = VsAgentDoubles(
            **kwargs,
            log_path=str(ROOT / "data" / "ladder-games-doubles.log"),
        )
    else:
        agent = VsAgent(
            **kwargs,
            log_path=str(ROOT / "data" / "ladder-games.log"),
            brain={"model": "v2_model.txt", "featurizer": "v2", "pi2": True},
        )

    seen, dumped = set(), set()

    async def announce() -> None:
        while True:
            for tag, b in agent.battles.items():
                if tag not in seen:
                    seen.add(tag)
                    room = tag if tag.startswith("battle-") else f"battle-{tag}"
                    print(f"BATTLE {room} https://play.pokemonshowdown.com/{room}", flush=True)
                    # no more infinite stalls: main has no auto-timer in randbats
                    await agent.ps_client.send_message("/timer on", room=room)
                if b.finished and tag not in dumped:
                    dumped.add(tag)  # per-game dump: kills never lose games
                    dump_battle(b)
                    result = "WIN" if b.won else "LOSS" if b.lost else "TIE"
                    print(f"{tag}: {result} (turns: {b.turn})", flush=True)
            await asyncio.sleep(2)

    if args.accept is not None:
        # only accept challenges matching --format: a mismatched format wedges
        # the client (format gate in _create_battle) into a reconnect loop
        fmt = args.format

        async def _filtered(self, split_message):
            challenger = split_message[2].strip()
            if (challenger != self.username and len(split_message) >= 6
                    and split_message[5] == fmt):
                await self._challenge_queue.put(challenger)

        import types
        agent._handle_challenge_request = types.MethodType(_filtered, agent)

    announcer = asyncio.create_task(announce())
    try:
        if args.challenge:
            print(f"challenging {args.challenge} ({args.format})...", flush=True)
            await agent.send_challenges(args.challenge, 1)
        elif args.accept is not None:
            who = args.accept or None
            print(f"waiting for challenge from {who or 'anyone'} ({args.format})...", flush=True)
            await agent.accept_challenges(who, 1)
        else:
            print(f"laddering as {creds['username']} for {args.games} games [{args.format}]...", flush=True)
            await agent.ladder(args.games)
    finally:
        announcer.cancel()

    for tag, b in agent.battles.items():  # anything finished but not yet dumped
        if tag not in dumped and b.finished:
            dump_battle(b)
            result = "WIN" if b.won else "LOSS" if b.lost else "TIE"
            print(f"{tag}: {result} (turns: {b.turn})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
