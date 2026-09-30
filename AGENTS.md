# AGENTS.md - showdown-brain

Battle AI for Pokemon Showdown **gen 9 random singles**, learned from public ladder replays.

## Why random battles

Hidden information is constrained: every species rolls from the known pool in
`~/pokemon-data/randbats-sets.json`. Unrevealed sets are enumerable — this is the
project's core advantage over free-teambuild formats.

## Architecture (milestones)

- **M1 (done)**: harvester + protocol parser -> `(state, action, outcome)` rows;
  evaluator V(s) = LightGBM win predictor (AUC 0.64, calibrated mid-range; upgrades:
  revealed-set belief features, more replays). Rows are per-decision-point, both players.
- **M2 (partial)**: policy clone pi(a|s) — flat multiclass stalls at ~35% top-1.
  Redesign as 2-stage (move-vs-switch classifier, then action ranker) before
  trusting it in search. Files: scripts/train_pi.py, data/pi_model.txt.
- **M3 (v0 done)**: kingambit plays! `scripts/selfplay.py` runs VsAgent (one-ply
  expectimax, V(s) leaves, pessimistic opponent = max-damage response) vs
  SimpleHeuristicsPlayer (poryclone) on the local server. First full game:
  35 turns, coherent setup/pivot patterns, LOSS. Decisions: data/selfplay.log.
  Next: pi-weighted opponent model, depth-2, tera, protect scouting.
- **M4 (stretch)**: self-play fine-tuning.

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
uv run python -m sb.parser
uv run python scripts/train_v.py     # -> models/v_model.txt
uv run python scripts/train_pi2.py   # -> models/pi2_*.txt
uv run python scripts/selfplay.py    # bot vs baseline on a local server
uv run python scripts/human.py       # bot accepts one human challenge
```

## Gotchas

- LightGBM spawns a thread per core by default — training scripts cap
  `num_threads: 4` and should be launched `nice -n 15`. Load average hit 98
  uncapped. Don't remove the cap.
- `pkill -f <pattern>` from a shell whose command line contains the pattern
  kills the shell itself — use `[b]racket` patterns, and put kills and starts
  in separate tool calls.
- Replay archive may rate-limit (`search.json` timeouts) after big harvests;
  back off and resume — harvest.py checkpoints via manifest.
