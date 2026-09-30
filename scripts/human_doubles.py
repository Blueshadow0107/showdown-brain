"""human_doubles.py — kingambit (VsAgentDoubles) accepts ONE doubles challenge."""
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402
from poke_env import AccountConfiguration, ServerConfiguration  # noqa: E402
import poke_env.ps_client.ps_client as psc  # noqa: E402

from sb.agent_doubles import VsAgentDoubles  # noqa: E402

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


async def main():
    import os
    prefix = "SHOWDOWN_BOT_"
    user, pw = os.environ.get(prefix + "USER"), os.environ.get(prefix + "PASS")
    if user and pw:
        account = AccountConfiguration(user, pw)
    else:
        a = json.load(open(ROOT / "data" / "bot-account.json"))
        account = AccountConfiguration(a["username"], a["password"])
    king = VsAgentDoubles(
        account_configuration=account,
        server_configuration=LOCAL,
        log_path=str(ROOT / "data" / "human-games-doubles.log"),
    )
    print("kingambit is online — waiting for a DOUBLES challenge (anyone, 1 game)", flush=True)
    await king.accept_challenges(None, 1)
    for tag, b in king.battles.items():
        result = "WIN" if b.won else "LOSS" if b.lost else "TIE"
        print(f"{tag}: {result} (turns: {b.turn})", flush=True)
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
