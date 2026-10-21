"""local_ps_login.py — monkeypatch poke-env auth for the private sim server.

LocalhostServerConfiguration points authentication at PS Main; the local sim
drains POST bodies before its customhttpresponse handler runs, so login GETs
action.php with the credentials in the query string instead. install() swaps
PSClient.log_in for that GET-based flow; call it before connecting any client
to ws://localhost:8000.
"""
import json

import requests
import poke_env.ps_client.ps_client as psc


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


def install() -> None:
    """Route PSClient.log_in through the local action.php. Idempotent."""
    psc.PSClient.log_in = local_login
