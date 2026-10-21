"""ffa_selfplay.py — four (or more) of our accounts fill a gen-9 random FFA
table and play each other, every seat with its own brain. Every game is
dumped for the learning loop.

    uv run python scripts/ffa_selfplay.py --games 10 --accounts bot,baseline,ladder,rando

Challenges are 1v1-only, so each seat enters the matchmaker queue for
gen9freeforallrandombattle and the server groups four searchers into one
room. Seat brains rotate through [v2, v1-classic, v_ffa (or v2), random].
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from poke_env import AccountConfiguration, ServerConfiguration  # noqa: E402
from poke_env.concurrency import handle_threaded_coroutines  # noqa: E402
from poke_env.player import Player  # noqa: E402

import sb.ffa_poke_env  # noqa: E402
import sb.local_ps_login  # noqa: E402
from sb.agent_ffa import VsAgentFFA  # noqa: E402
from sb.ffa_poke_env import FFABattle  # noqa: E402
from sb.learn import dump_battle  # noqa: E402

LOCAL = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/~~localhost/action.php?",
)

# The private sim's matchmaker refuses to group searchers sharing a source IP
# (server/ladders.ts matchmakingOK, Config.noipchecks = false). All clients in
# this process share one poke-env loop, so per-seat binding is done by
# destination: each seat targets a distinct loopback alias, and
# install_loopback_bind() pins the socket's local address to that same alias.
# (Without the explicit bind the kernel routes 127.0.0.N with src 127.0.0.1;
#  real interface addresses don't all work either — host->docker0 traffic gets
#  SNATed to the LAN IP by local netfilter rules.)
SEAT_HOSTS = ("127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4")


def seat_config(i: int) -> ServerConfiguration:
    host = SEAT_HOSTS[i % len(SEAT_HOSTS)]
    return ServerConfiguration(f"ws://{host}:8000/showdown/websocket",
                               LOCAL.authentication_url)


_LOOPBACK_HOSTS = frozenset(SEAT_HOSTS)
_orig_create_connection = asyncio.BaseEventLoop.create_connection


async def _bound_create_connection(self, protocol_factory, host=None,
                                   port=None, **kw):
    if host in _LOOPBACK_HOSTS and kw.get("local_addr") is None:
        kw["local_addr"] = (host, 0)
    return await _orig_create_connection(self, protocol_factory, host, port,
                                         **kw)


def install_loopback_bind() -> None:
    """Pin client sockets to their loopback alias so the server sees distinct
    source IPs per seat. Idempotent."""
    if asyncio.BaseEventLoop.create_connection is not _bound_create_connection:
        asyncio.BaseEventLoop.create_connection = _bound_create_connection


FFA_FORMAT = "gen9freeforallrandombattle"
STALL_SECONDS = 900


def load_account(name: str) -> AccountConfiguration | None:
    user = os.environ.get(f"SHOWDOWN_{name.upper()}_USER")
    pw = os.environ.get(f"SHOWDOWN_{name.upper()}_PASS")
    if user and pw:
        return AccountConfiguration(user, pw)
    path = ROOT / "data" / f"{name}-account.json"
    if path.exists():
        a = json.loads(path.read_text())
        return AccountConfiguration(a["username"], a["password"])
    return None


def seat_mix() -> list:
    """Per-seat brains, rotating. None = random mover."""
    ffa_model = "v_ffa.txt" if (ROOT / "models" / "v_ffa.txt").exists() \
        else "v2_model.txt"
    return [
        {"model": "v2_model.txt", "featurizer": "v2", "pi2": True},
        {"model": "history/v1_classic.txt", "featurizer": "v1-classic", "pi2": False},
        {"model": ffa_model, "featurizer": "v2", "pi2": True},
        None,
    ]


class FFASeat(Player):
    """One seat of the FFA table: plays every battle it is placed in with a
    fixed brain (None = random mover)."""

    def __init__(self, *args, brain=None, log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # the brain's own client is inert (never started) — same delegate
        # pattern as scripts/serve.py; distinct throwaway accounts per seat.
        self._brain = VsAgentFFA(
            account_configuration=AccountConfiguration(
                f"unused_ffa_{id(self)}", "unused"),
            server_configuration=LOCAL,
            battle_format=FFA_FORMAT,
            log_path=log_path,
            brain=brain,
        ) if brain else None

    def choose_move(self, battle):
        if self._brain is not None and isinstance(battle, FFABattle):
            return self._brain.choose_move(battle)
        return self.choose_random_move(battle)

    def teampreview(self, battle):
        if self._brain is not None and isinstance(battle, FFABattle):
            return self._brain.teampreview(battle)
        return super().teampreview(battle)


async def run_seat(seat: FFASeat, ctl: dict) -> None:
    """Login once, then keep the seat in the FFA queue until told to stop."""
    await handle_threaded_coroutines(seat.ps_client.wait_for_login(),
                                     seat.ps_client.loop)
    queued = False
    while not ctl["stop"]:
        in_battle = any(not b.finished for b in seat.battles.values())
        if in_battle:
            queued = False
        elif not queued:
            try:
                await handle_threaded_coroutines(
                    seat.ps_client.search_ladder_game(FFA_FORMAT, None),
                    seat.ps_client.loop)
                queued = True
            except Exception as e:
                print(f"{seat.username}: search failed: {e!r}", flush=True)
        await asyncio.sleep(2)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1,
                    help="FFA games to complete across the group before exiting")
    ap.add_argument("--accounts", default="bot,baseline,ladder",
                    help="comma-separated account names; credentials from "
                         "SHOWDOWN_<NAME>_USER/_PASS or data/<name>-account.json")
    ap.add_argument("--eog", type=int, default=1,
                    help="extra games allowed to finish past --games before a hard stop")
    args = ap.parse_args()

    sb.ffa_poke_env.install()
    sb.local_ps_login.install()
    install_loopback_bind()

    accounts = {}
    for name in [n.strip() for n in args.accounts.split(",") if n.strip()]:
        acc = load_account(name)
        if acc is None:
            print(f"warning: no credentials for '{name}' "
                  f"(SHOWDOWN_{name.upper()}_USER/_PASS or data/{name}-account.json)",
                  flush=True)
        else:
            accounts[name] = acc
    if len(accounts) < 4:
        print(f"error: FFA needs at least 4 seats, only {len(accounts)} resolvable "
              f"({', '.join(accounts) or 'none'})", file=sys.stderr, flush=True)
        return 2

    mix = seat_mix()
    seats = []
    for i, (name, acc) in enumerate(accounts.items()):
        seats.append(FFASeat(
            account_configuration=acc,
            server_configuration=seat_config(i),
            battle_format=FFA_FORMAT,
            brain=mix[i % len(mix)],
            log_path=str(ROOT / "data" / f"ffa_selfplay_{name}.log"),
        ))

    ctl = {"stop": False}
    tasks = [asyncio.create_task(run_seat(seat, ctl)) for seat in seats]
    finished = {}  # battle tag -> (seat, battle); first observer wins
    seen_active = set()
    last_activity = time.monotonic()
    hard_cap = args.games + args.eog
    exit_code = 1

    print(f"ffa selfplay: {len(seats)} seats, target {args.games} game(s), "
          f"hard stop at {hard_cap}", flush=True)
    try:
        while True:
            for seat in seats:
                for tag, b in seat.battles.items():
                    if tag not in seen_active and not b.finished:
                        seen_active.add(tag)
                        last_activity = time.monotonic()
                    if isinstance(b, FFABattle) and b.finished and tag not in finished:
                        finished[tag] = (seat, b)
                        last_activity = time.monotonic()
                        dump_battle(b)
                        winner = getattr(b, "winner_username", None) or "?"
                        print(f"game {len(finished)}/{args.games}: {tag} "
                              f"winner={winner} (turns: {b.turn})", flush=True)
            if len(finished) >= args.games:
                exit_code = 0
                ctl["stop"] = True
            in_flight = any(
                isinstance(b, FFABattle) and not b.finished
                for seat in seats for b in seat.battles.values())
            live_tasks = sum(1 for t in tasks if not t.done())
            if len(finished) >= hard_cap:
                print(f"hard stop: {len(finished)} games reached", flush=True)
                break
            if ctl["stop"] and (not in_flight or live_tasks == 0):
                break
            if live_tasks == 0:
                print("error: all seats died before the target", flush=True)
                break
            if time.monotonic() - last_activity > STALL_SECONDS:
                print(f"error: no game activity for {STALL_SECONDS}s", flush=True)
                break
            await asyncio.sleep(2)
    finally:
        ctl["stop"] = True

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for seat in seats:
        try:
            await handle_threaded_coroutines(
                seat.ps_client.send_message("/cancelsearch"), seat.ps_client.loop)
        except Exception:
            pass
    for seat, result in zip(seats, results):
        if isinstance(result, Exception):
            print(f"seat {seat.username} crashed: {result!r}", flush=True)
    winners = {}
    for _tag, (_seat, b) in finished.items():
        winner = getattr(b, "winner_username", None) or "?"
        winners[winner] = winners.get(winner, 0) + 1
    print(f"done: {len(finished)} ffa games; winners: {winners}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
