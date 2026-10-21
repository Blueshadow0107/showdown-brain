# AGENTS.md - showdown-brain

Battle AI for Pokemon Showdown **gen 9 random battles (singles + doubles)**, learned from public ladder replays.

## Why random battles

Hidden information is constrained: every species rolls from the known pool in
`~/pokemon-data/randbats-sets.json`. Unrevealed sets are enumerable — this is the
project's core advantage over free-teambuild formats.

## Status (2026-10; layout and data flow: ARCHITECTURE.md)

- **V(s)** — LightGBM win predictor; current build = `models/v2_model.txt` with the v2 featurizer (393 dims), AUC 0.72 on ~202k rows (`models/v2_metrics.json`; ~2.3k are our own games). Doubles has its own `v_model_doubles.txt`.
- **π** — `pi2` (move-vs-switch, then action ranker) weights the opponent's response; `pi2doubles` for doubles. The flat multiclass clone (`train_pi.py`, ~35% top-1) is superseded.
- **Player** — `VsAgent` / `VsAgentDoubles`: one-ply expectimax with V(s) leaves; every tuning constant lives in `sb/heuristics.py` (nowhere else). Physics lives in `sb/transitions.py`, nothing learned.
- **Live** — `scripts/serve.py` is the lobby bot (singles + doubles; systemd user service, toggled from the waybar services panel). `scripts/gauntlet.py` plays past selves and `sb.learn` dumps our own games back into the corpus for retraining.
- **FFA** — `[Gen 9] Free-For-All Random Battle` (`gen9freeforallrandombattle`, 6 mons per player × 4). `FFABattle` (sb/ffa_poke_env.py) patches poke-env for 4 sides; `VsAgentFFA` (sb/agent_ffa.py) projects each decision into one 1v1 state per live opponent, scores them with the v2 featurizer + a V model, and aggregates with the `ffa_*` constants in heuristics.py. `sb/parser_ffa.py` turns FFA logs into pairwise v2 rows (one per live opponent per decision, `outcome` = did the actor win the FFA); `models/v_ffa.txt` is the FFA-trained V. Data: public harvest (`--format gen9freeforallrandombattle`, never rating-filter — FFA replays are unrated), `scripts/ffa_selfplay.py` (4 local accounts), human games via learn.py.
- **Ideas not yet done** (from earlier notes): depth-2 search, tera / protect scouting, self-play fine-tuning.

## Bot plumbing (hard-won, do not rediscover)

- poke-env 0.16 auth: monkeypatched in scripts/selfplay.py — LocalhostServerConfiguration
  points auth at PS Main; we GET our local action.php instead (POST body is drained
  by the sim's static handler before customhttpresponse runs).
- Wire format: messages are `|<cmd>` (leading pipe, empty room). Bare commands with
  no pipe are dropped by the sim.
- VsAgent: src/sb/agent.py; state mapping sb/showdown_state.py (poke-env 0.16
  weather/fields/side_conditions are DICTS keyed by enum, not enums).
- Transitions sb/transitions.py use our own damage formula (hypothetical-friendly);
  poke_env.calc.calculate_damage is exact but live-state only — validate offline.
- FFA (freeforall) — poke-env has no 4-side support: `FFABattle` + `install()`
  in sb/ffa_poke_env.py handle it (title/player events, per-side buckets,
  elimination). Target a foe with `move <i> <loc>` where loc comes from
  sb/ffa.target_loc (self-loc is −1 for even sides, −2 for odd; ±1/±2 cover
  the three foes). Slot letters vary by side (p1a/p2a/p3b/p4b).
- The private sim's matchmaker requires pairwise-distinct source IPs and
  `periodicMatch` skips playerCount>2 formats — FFA self-play binds each seat
  to a 127.0.0.x loopback alias (see SEAT_HOSTS in scripts/ffa_selfplay.py).
- The private sim signs all logins as unregistered (config.js makeAssertion),
  so /pm is server-side blocked — the "ffa on"/"ffa off" PM trigger only works
  where accounts are registered (e.g. PS Main). Locally the bot instead
  stands in the FFA queue whenever data/ffa-standing exists (or
  SHOWDOWN_FFA_STANDING=1); delete the file to stand down, no restart needed.

## Conventions

- Protocol parsing lives in `src/sb/parser.py`; it tracks showdown's replay log format
  (`|switch|`, `|-boost|`, `|-sidestart|`...). Only common events are handled —
  extend the dispatch table, don't regex the whole line.
- State dicts are the canonical intermediate form; `features.py` owns the numeric vector.
  Keep both perspective-symmetric: rows store `my`/`foe` from the acting player's view.
- Game data is VENDORED in `assets/` (copied from ~/pokemon-data — never re-download
  what `assets/game_data.json` already has). Resolution: `$POKEMON_DATA` env →
  repo `assets/` → `~/pokemon-data`.
- Trained models live in `models/` (git-tracked, small). Training scripts write
  there; `agent.py` reads `models/v_model.txt`.
- Bot credentials: `SHOWDOWN_BOT_USER/_PASS` (and `SHOWDOWN_BASELINE_*`) env vars,
  falling back to gitignored `data/{bot,baseline}-account.json`. Never commit creds.
- Split train/eval by **game id**, never by row — rows from one game correlate.
- No API keys in this repo. Jev/TypeSafe stays out of the live loop.

## Commands

```bash
uv sync
uv run python -m sb.harvest --limit 200 --rating-min 1400
uv run python -m sb.harvest --format gen9freeforallrandombattle --limit 400  # FFA: never rating-filter
uv run python -m sb.parser         # -> data/rows_v2.jsonl (v2 states)
uv run python -m sb.parser_doubles # -> data/rows_v2_doubles.jsonl
uv run python -m sb.parser_ffa     # -> data/rows_ffa.jsonl (pairwise FFA rows)
uv run python scripts/train_v_v2.py     # -> models/v2_model.txt
uv run python scripts/train_v_ffa.py    # -> models/v_ffa.txt
uv run python scripts/train_pi2.py      # -> data/pi2_*
uv run python scripts/gauntlet.py      # self-gauntlet + online learning
uv run python scripts/ffa_selfplay.py --games 10 --accounts bot,baseline,ladder,ffa4  # FFA data gen
uv run python scripts/ladder.py        # ladder runs with per-game dumps
uv run python scripts/serve.py         # THE lobby bot (singles + doubles + ffa)
```

See ARCHITECTURE.md for the system layout and contribution rules.

## Gotchas

- LightGBM spawns a thread per core by default — training scripts cap
  `num_threads: 4` and should be launched `nice -n 15`. Load average hit 98
  uncapped. Don't remove the cap.
- Never `pkill -f <pattern>` when your own command line contains the pattern — it
  kills the shell. Kill by pid (or use `[b]racket` patterns in a separate call).
- Replay archive may rate-limit (`search.json` timeouts) after big harvests;
  back off and resume — harvest.py checkpoints via manifest.
