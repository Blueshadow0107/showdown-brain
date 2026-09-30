"""selfplay_doubles.py — kingambit (VsAgentDoubles) vs random-move baseline
(RandomDoublesPlayer) on the local server, gen9randomdoublesbattle."""
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from poke_env import AccountConfiguration, ServerConfiguration  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402
from poke_env.player import Player  # noqa: E402
import poke_env.ps_client.ps_client as psc  # noqa: E402

from sb.agent_doubles import VsAgentDoubles  # noqa: E402

LOCAL = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/~~localhost/action.php?",
)


async def local_login(self, split_message):
    """PSClient.log_in patched for our local server: GET instead of POST
    (our action.php emulation reads query params; the sim's static handler
    drains POST bodies before customhttpresponse runs)."""
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


def acc(kind: str) -> AccountConfiguration:
    """Credentials: SHOWDOWN_{KIND}_USER / _PASS env vars, else data/{kind}-account.json."""
    prefix = f"SHOWDOWN_{kind.upper()}_"
    import os
    user, pw = os.environ.get(prefix + "USER"), os.environ.get(prefix + "PASS")
    if user and pw:
        return AccountConfiguration(user, pw)
    a = json.load(open(ROOT / "data" / f"{kind}-account.json"))
    return AccountConfiguration(a["username"], a["password"])


class RandomDoublesPlayer(Player):
    """Minimal baseline: uniformly random legal doubles order."""

    def choose_move(self, battle):
        return self.choose_random_doubles_move(battle)


async def main():
    bot_acc, base_acc = acc("bot"), acc("baseline")
    king = VsAgentDoubles(account_configuration=bot_acc, server_configuration=LOCAL,
                          battle_format="gen9randomdoublesbattle",
                          start_timer_on_battle_start=True,
                          log_path=str(ROOT / "data" / "selfplay_doubles.log"))
    base = RandomDoublesPlayer(account_configuration=base_acc,
                               server_configuration=LOCAL,
                               battle_format="gen9randomdoublesbattle",
                               start_timer_on_battle_start=True)
    print("both players up — challenging", flush=True)
    await asyncio.gather(
        base.send_challenges(to_id_str(bot_acc.username), 1),
        king.accept_challenges(to_id_str(base_acc.username), 1),
    )
    for tag, b in king.battles.items():
        result = "WIN" if b.won else "LOSS" if b.lost else "TIE"
        print(f"{tag}: {result} (turns: {b.turn})", flush=True)
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
