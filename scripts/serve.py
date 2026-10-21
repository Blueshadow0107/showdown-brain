"""serve.py — kingambit accepts singles, doubles and free-for-all battles.

One account, one process: dispatches each battle to the right brain by
battle type. Random-battle formats generate teams server-side, so a single
listener can serve every format. FFA has no challenge flow: the bot queues
into gen9freeforallrandombattle as the group's standing 4th player. The
standing switch is server-dependent: on the LOCAL sim it is data/ffa-standing
(or SHOWDOWN_FFA_STANDING=1) — delete the file to stand down. On MAIN the
file is ignored (it would hog both battle slots and starve challenge
accepts); queue there only via SHOWDOWN_FFA_STANDING=1 or the "ffa on" PM,
and stand down with "ffa off".

Server selection: SHOWDOWN_SERVER=local (default) plays on the private sim;
SHOWDOWN_SERVER=main plays on play.pokemonshowdown.com with the ladder
account (stock poke-env login — no local patch) and turns battle timers on.
"""
import asyncio
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from poke_env import AccountConfiguration, ServerConfiguration  # noqa: E402
from poke_env import ShowdownServerConfiguration  # noqa: E402
from poke_env.battle import DoubleBattle  # noqa: E402
from poke_env.concurrency import handle_threaded_coroutines  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402
from poke_env.player import Player  # noqa: E402

import sb.ffa_poke_env  # noqa: E402
import sb.local_ps_login  # noqa: E402
from sb.agent import VsAgent  # noqa: E402
from sb.agent_doubles import VsAgentDoubles  # noqa: E402
from sb.agent_ffa import VsAgentFFA  # noqa: E402
from sb.ffa_poke_env import FFABattle  # noqa: E402
from sb.learn import dump_battle, learn_new_games  # noqa: E402

MAIN = os.environ.get("SHOWDOWN_SERVER", "local").lower() == "main"
LOCAL = ShowdownServerConfiguration if MAIN else ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/~~localhost/action.php?",
)

FFA_FORMAT = "gen9freeforallrandombattle"
ADOPTABLE_FORMATS = {"gen9randombattle", "gen9randomdoublesbattle", FFA_FORMAT}
FFA_STANDING_FILE = ROOT / "data" / "ffa-standing"
# Only these accounts get their friend requests auto-accepted (userids).
FRIEND_ALLOW = {to_id_str(u) for u in
                os.environ.get("SHOWDOWN_FRIENDS", "cyanidepopcorn").split(",") if u.strip()}


def ffa_standing_env() -> bool:
    return os.environ.get("SHOWDOWN_FFA_STANDING", "").lower() in \
        {"1", "true", "yes", "on"}


def ffa_standing_requested() -> bool:
    # The standing file is the LOCAL group's "standing 4th" switch; on main
    # its mere existence must not queue public FFA. There the queue is
    # env/PM-only, so the PM toggle is never stomped by the file watcher.
    return ffa_standing_env() if MAIN else \
        ffa_standing_env() or FFA_STANDING_FILE.exists()


class UniversalAgent(Player):
    """Routes each battle to a singles, doubles or FFA brain. The delegates
    are never started as players — only their decision methods are used."""

    def __init__(self, *args, log_path=None, **kwargs):
        # Message handling can start before the delegates below exist — a
        # battle message raced __init__ once ('_ffa' AttributeError). Default
        # to None and guard the decision methods until they are built.
        self._singles = None
        self._doubles = None
        self._ffa = None
        super().__init__(*args, **kwargs)
        # delegates share our connection for orders; their own clients are
        # inert (never started). Distinct throwaway accounts avoid clashes.
        self._singles = VsAgent(
            account_configuration=AccountConfiguration("unused_s", "unused"),
            server_configuration=LOCAL,
            log_path=str(ROOT / "data" / "human-games.log"),
            brain={"model": "v2_model.txt", "featurizer": "v2", "pi2": True},
        )
        self._doubles = VsAgentDoubles(
            account_configuration=AccountConfiguration("unused_d", "unused"),
            server_configuration=LOCAL,
            battle_format="gen9randomdoublesbattle",
            log_path=str(ROOT / "data" / "human-games-doubles.log"),
        )
        ffa_model = "v_ffa.txt" if (ROOT / "models" / "v_ffa.txt").exists() \
            else "v2_model.txt"
        self._ffa = VsAgentFFA(
            account_configuration=AccountConfiguration("unused_f", "unused"),
            server_configuration=LOCAL,
            battle_format=FFA_FORMAT,
            log_path=str(ROOT / "data" / "human-games-ffa.log"),
            brain={"model": ffa_model, "featurizer": "v2", "pi2": True},
        )
        self._ffa_enabled = ffa_standing_requested()
        self._ffa_search_issued = False
        self._hook_private_messages()

    def _hook_private_messages(self):
        """poke-env 0.16 only logs PM text — there is no listener hook, so
        wrap the client's message handler: act on exact PM commands first,
        then defer to the original handler (challenge routing, logging)."""
        agent = self
        original = self.ps_client._handle_message

        async def _handle_with_pm(message: str):
            try:
                for line in message.split("\n"):
                    parts = line.split("|")
                    if len(parts) >= 5 and parts[1] == "pm":
                        await agent._on_pm(parts[2].strip(), parts[4])
            except Exception as e:
                agent.logger.warning("pm hook failed: %r", e)
            try:
                await original(message)
            except Exception as e:
                # Challenges arrive as PMs — a parse crash must not take the
                # message that carries them.
                agent.logger.warning("pm handler failed: %r", e)

        self.ps_client._handle_message = _handle_with_pm

    async def _on_pm(self, sender: str, text: str):
        if to_id_str(sender) == to_id_str(self.username):
            return
        if "sent you a friend request!" in text:
            if to_id_str(sender) in FRIEND_ALLOW:
                await self.ps_client.send_message(
                    f"/friends accept {to_id_str(sender)}")
                self.logger.info("accepted friend request from %s", sender)
            else:
                self.logger.info("ignoring friend request from %s", sender)
        elif text == "ffa on":
            self._ffa_enabled = True
            self._ffa_search_issued = False
            await self.ps_client.send_message(
                f"/pm {sender}, ffa queue on — searching {FFA_FORMAT}")
        elif text == "ffa off":
            self._ffa_enabled = False
            self._ffa_search_issued = False
            # poke-env 0.16 has no search-cancellation API; /cancelsearch is
            # the server's own command and cancels every queued search.
            await self.ps_client.send_message("/cancelsearch")
            await self.ps_client.send_message(f"/pm {sender}, ffa queue off")

    async def _handle_challenge_request(self, split_message):
        """poke-env drops incoming challenges whose format != self._format
        (player.py:428) and _create_battle repeats the check at battle init
        (player.py:192). A universal listener adopts the challenge's format
        so both gates pass — but only for formats it actually plays."""
        challenger = split_message[2].strip()
        if challenger != self.username:
            if len(split_message) >= 6 and split_message[5] in ADOPTABLE_FORMATS:
                self._format = split_message[5]
                await self._challenge_queue.put(challenger)
            else:
                fmt = split_message[5] if len(split_message) >= 6 else "?"
                self.logger.info(
                    "ignoring challenge from %s (%s)", challenger, fmt)

    def refresh_ffa_standing(self):
        """The standing flag is a file/env check so it toggles without a
        restart; the watcher re-reads it on every poll."""
        if ffa_standing_requested():
            self._ffa_enabled = True

    async def maybe_search_ffa(self):
        """Re-issue the FFA ladder search whenever enabled and idle, so the
        bot stays queued as the group's standing 4th player."""
        if not self._ffa_enabled or self._ffa_search_issued:
            return
        if any(isinstance(b, FFABattle) and not b.finished
               for b in self.battles.values()):
            return
        if not self._challenge_queue.empty():
            # A challenge is waiting on a battle slot; flipping _format to
            # FFA now would make poke-env drop the challenge's battle at
            # init (player._create_battle gates on self.format).
            return

        async def _search():
            await self.ps_client.wait_for_login()
            self._format = FFA_FORMAT
            # Set BEFORE issuing: otherwise every 5s poll re-sends /search
            # ("already searching" spam -> sim rate-limit -> late games).
            self._ffa_search_issued = True
            try:
                await self.ps_client.search_ladder_game(FFA_FORMAT, None)
            except Exception:
                self._ffa_search_issued = False
                raise

        await handle_threaded_coroutines(_search(), self.ps_client.loop)

    def choose_move(self, battle):
        if self._ffa is None:  # delegates not built yet — see __init__
            return self.choose_default_move()
        if isinstance(battle, FFABattle):
            return self._ffa.choose_move(battle)
        if isinstance(battle, DoubleBattle):
            return self._doubles.choose_move(battle)
        return self._singles.choose_move(battle)

    def teampreview(self, battle):
        if self._ffa is None:
            return super().teampreview(battle)
        if isinstance(battle, FFABattle):
            return self._ffa.teampreview(battle)
        return super().teampreview(battle)


async def accept_loop(king: UniversalAgent):
    while True:
        try:
            await king.accept_challenges(None, 1)
        except Exception as e:
            print(f"  accept failed (non-fatal): {e!r}", flush=True)
            await asyncio.sleep(3)


async def ffa_watch(king: UniversalAgent):
    while True:
        try:
            king.refresh_ffa_standing()
            await king.maybe_search_ffa()
        except Exception as e:
            print(f"  ffa search failed (non-fatal): {e!r}", flush=True)
        await asyncio.sleep(5)


async def main():
    sb.ffa_poke_env.install()
    if not MAIN:
        sb.local_ps_login.install()

    if MAIN:
        a = json.load(open(ROOT / "data" / "ladder-account.json"))
        user = os.environ.get("SHOWDOWN_LADDER_USER") or a["username"]
        pw = os.environ.get("SHOWDOWN_LADDER_PASS") or a["password"]
        account = AccountConfiguration(user, pw)
    else:
        prefix = "SHOWDOWN_BOT_"
        user = os.environ.get(prefix + "USER")
        pw = os.environ.get(prefix + "PASS")
        if user and pw:
            account = AccountConfiguration(user, pw)
        else:
            a = json.load(open(ROOT / "data" / "bot-account.json"))
            account = AccountConfiguration(a["username"], a["password"])
    king = UniversalAgent(
        account_configuration=account,
        server_configuration=LOCAL,
        battle_format="gen9randombattle",
        max_concurrent_battles=2,
    )
    await king.ps_client.send_message("/join lobby")
    where = "PS MAIN" if MAIN else "lobby"
    standing = "on (standing 4th — rm data/ffa-standing to stand down)" \
        if king._ffa_enabled else "off (PM 'ffa on' to queue)"
    print(f"{king.username} is online ({where}) — accepting singles, doubles "
          f"and FFA; ffa queue {standing}", flush=True)
    # A request sent while the bot was offline never arrives as a PM but is
    # still pending server-side; accepting a non-pending id is a harmless no-op.
    for friend in sorted(FRIEND_ALLOW):
        await king.ps_client.send_message(f"/friends accept {friend}")
    asyncio.create_task(accept_loop(king))
    asyncio.create_task(ffa_watch(king))
    seen, started = set(), set()
    while True:
        for tag, b in king.battles.items():
            if tag not in started:
                started.add(tag)
                if MAIN:  # main randbats has no auto-timer
                    await king.ps_client.send_message("/timer on", room=tag)
            if tag in seen or not b.finished:
                continue
            seen.add(tag)
            dump_battle(b)  # every game becomes training data
            result = "WIN" if b.won else "LOSS" if b.lost else "TIE"
            if isinstance(b, FFABattle):
                kind = "ffa"
                king._ffa_search_issued = False  # re-queue as the 4th
            elif isinstance(b, DoubleBattle):
                kind = "doubles"
            else:
                kind = "singles"
            print(f"{tag} [{kind}]: {result} (turns: {b.turn})", flush=True)
        # fold finished games into the personal corpus + rebuild the model
        try:
            n = await asyncio.get_event_loop().run_in_executor(None, learn_new_games)
            if n:
                print(f"  learned from {n} games", flush=True)
        except Exception as e:
            print(f"  learn failed (non-fatal): {e!r}", flush=True)
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
