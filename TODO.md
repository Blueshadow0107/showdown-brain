# TODO — waybar bot → Pokémon Showdown Main

**Status:** parked until Monday 2026-10-05 (user's request limits reset).
**Goal:** make the waybar services-panel toggle run the bot against the REAL
showdown server (play.pokemonshowdown.com) so it can ladder — user does NOT
want the local-server connection anymore.

## Context (diagnosed 2026-10-03)

- Waybar button works fine — it toggles `showdown-brain.service`
  (`~/.config/systemd/user/showdown-brain.service`), which runs
  `scripts/serve.py`. The "it doesn't work" feeling was: (a) a stale local
  listener competing for the login, (b) `serve.py` hardcodes localhost:8000.
- **Stale local instance still running:** PID 439627 (`uv run python
  scripts/serve.py`, setsid-nohup'd 2026-10-01 05:05, parent bash 439626).
  It holds the `kingambit` name on the local server. Kill it when doing this
  work (`kill 439627 439634`) — or `pkill -f "[s]erve.py"`.
- `scripts/ladder.py` already knows how to talk to Main: passes
  `ShowdownServerConfiguration` explicitly (`wss://sim3.psim.us/showdown/websocket`,
  auth `https://play.pokemonshowdown.com/action.php?`), uses
  `data/ladder-account.json` (username `kingambot-v2`), and its docstring says
  **stock poke-env login works on Main** (the localhost monkeypatch in
  serve.py is local-only; main uses RSA-SHA1, local RSA-SHA256).

## Plan

1. `serve.py`: add `--server {main,local}` (default `main`, env override
   `SHOWDOWN_SERVER`). On main: use `ShowdownServerConfiguration`, do NOT
   apply the `local_login` monkeypatch, fall back to `ladder-account.json`
   for creds (keep `SHOWDOWN_BOT_USER/_PASS` env override + bot-account.json
   for local). Print the actual username, not the hardcoded "kingambit".
2. **Delegate bug (fix regardless):** the throwaway `VsAgent` delegates
   (`unused_s`/`unused_d`) get `start_listening=True` by default in poke-env
   0.16, so they connect + fail login (`KeyError: 'assertion'` spam). Pass
   `start_listening=False`. On Main this is mandatory — the errors would be
   publicly visible.
3. Add `--ladder` flag: extra asyncio task looping `king.ladder(1)` so the
   listener both accepts challenges AND queues rated games. On main there's
   no auto-timer: send `/timer on` per new battle (copy ladder.py) and print
   the spectate URL (`https://play.pokemonshowdown.com/<room>`).
4. Update the unit: `ExecStart=... uv run python scripts/serve.py --server main --ladder`, `daemon-reload`, restart, watch journal for
   `updateuser`/lobby join with no assertion errors.
5. Consider: notify via `notify-send` from the toggle when the service dies
   within a few seconds of start (waybar currently announces "started"
   unconditionally — the 13:53 start/stop churn was likely a double-click).
6. Optional: `systemctl --user enable` the service if the bot should survive
   reboots (currently disabled by design — waybar-toggled).

## Verify

- `journalctl --user -u showdown-brain -f` → login as `kingambot-v2`, lobby
  join, `/search` accepted (`|updatesearch`), battles start, per-game
  WIN/LOSS lines + "learned from N games".
- Challenge it on main from a human account to confirm accept-path still works.
