"""human_any.py — kingambit accepts BOTH singles and doubles challenges.

One account, one process: dispatches each battle to the right brain by
battle type. Random-battle formats generate teams server-side, so a single
listener can serve both formats.
"""
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from poke_env import AccountConfiguration, ServerConfiguration  # noqa: E402
from poke_env.battle import DoubleBattle  # noqa: E402
from poke_env.player import Player  # noqa: E402
import poke_env.ps_client.ps_client as psc  # noqa: E402

from sb.agent import VsAgent  # noqa: E402
from sb.agent_doubles import VsAgentDoubles  # noqa: E402
from sb.learn import dump_battle, learn_new_games  # noqa: E402

LOCAL = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/~~localhost/action.php?",
)


async def local_login(self, split_message):
    if self.account_configuration.password:
        r = requests.get(
            self.server_configuration.authentication_url,
            params={
                "act": "login",
                "name": self.account_configuration.username,
                "pass": self.account_configuration.password,
                "challstr": split_message[2] + "%7C" + split_message[3],
            },
            timeout=10.0,
        )
        assertion = json.loads(r.text[1:])["assertion"]
    else:
        assertion = ""
    await self.send_message(f"/trn {self.username},0,{assertion}")
    await self.change_avatar(self._avatar)


psc.PSClient.log_in = local_login


class UniversalAgent(Player):
    """Routes each battle to a singles or doubles brain. The delegates are
    never started as players — only their decision methods are used."""

    def __init__(self, *args, log_path=None, **kwargs):
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

    async def _handle_challenge_request(self, split_message):
        """poke-env drops incoming challenges whose format != self._format
        (player.py:428) and _create_battle repeats the check at battle init
        (player.py:192). A universal listener adopts the challenge's format
        so both gates pass for singles AND doubles."""
        challenger = split_message[2].strip()
        if challenger != self.username:
            if len(split_message) >= 6:
                self._format = split_message[5]
            await self._challenge_queue.put(challenger)

    def choose_move(self, battle):
        if isinstance(battle, DoubleBattle):
            return self._doubles.choose_move(battle)
        return self._singles.choose_move(battle)


async def main():
    import os
    prefix = "SHOWDOWN_BOT_"
    user, pw = os.environ.get(prefix + "USER"), os.environ.get(prefix + "PASS")
    if user and pw:
        account = AccountConfiguration(user, pw)
    else:
        a = json.load(open(ROOT / "data" / "bot-account.json"))
        account = AccountConfiguration(a["username"], a["password"])
    king = UniversalAgent(
        account_configuration=account,
        server_configuration=LOCAL,
        battle_format="gen9randombattle",
    )
    await king.ps_client.send_message("/join lobby")
    print("kingambit is online (lobby) — accepting singles AND doubles, forever", flush=True)
    seen = set()
    while True:
        await king.accept_challenges(None, 1)
        for tag, b in king.battles.items():
            if tag in seen:
                continue
            seen.add(tag)
            dump_battle(b)  # every game becomes training data
            result = "WIN" if b.won else "LOSS" if b.lost else "TIE"
            kind = "doubles" if isinstance(b, DoubleBattle) else "singles"
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
