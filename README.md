# showdown-brain

A battle AI for Pokémon Showdown **gen 9 random battles** — singles *and* doubles —
learned from public ladder replays. No hand-coded strategy: it learns a win
predictor V(s) and a policy prior π(a|s) from replays, then plays by one-ply
expectimax with a damage-formula transition model.

Train one on your own machine and battle it.

## How it plays

Every turn, for every legal action it can take:

```
EV(action) = V(state after my action + opponent's most likely response)
```

- **V(s)** — LightGBM win predictor (AUC 0.716) over a perspective-symmetric
  feature vector: species, stats, HP, boosts, status, item/ability semantic
  tags, hazards, weather, terrain, tera — plus a **momentum block** (speed
  control, both-side kill-clocks, priority presence, switch counters).
  Trained on ~200k decision points from public ladder replays.
- **Opponent model** — the opponent's response is scored with a gen-9 damage
  formula (stats, boosts, STAB, type matrix, weather/terrain, accuracy,
  effective speed) over the moves their species could be running, weighted by
  the π prior — so protect-heavy or switch-happy foes discount naturally.
- **π(a|s)** — two-stage policy clone: a move-vs-switch classifier (89% acc)
  plus move/switch rankers (41% top-1 / 78% top-3 over 310 moves) learned
  from what humans actually did in similar states.
- **Terastallization** — a first-class candidate whenever it's available,
  with tera typing handled in both offense and defense.
- **Tempo heuristics** — the search layer prices what a 1-ply value function
  can't: entry hazards on switch-in evals, a last-mon urgency penalty
  (no more setup-and-die), a repetition tax that breaks recovery-stall loops
  (roost ×10), and a switch-commit threshold so pivots must clear a real EV
  margin instead of bleeding tempo.

## Benchmarks

| judge | MAE | corr. with outcome | notes |
|---|---|---|---|
| V(s) | 0.26 | 0.92 | on 40 held-out mid-game states |
| Jev (TypeSafe System One) | 0.48 | 0.23 | semantic judge, same states |

The Jev comparison (`scripts/jev_benchmark.py`) was the fun experiment: a
calibrated decision model with full semantic knowledge of Pokémon *loses* to
the replay-learned V(s) at battle-state judgment — mostly because it ignores
fainted actives and compresses toward 0.5. Its edge shows only where V lacks
move-level knowledge. Scaffold included, API costs ~1¢ per 40 states.

Playing strength: ~50% win rate vs poke-env's SimpleHeuristics baseline in
random singles (10-game batches, zero decision errors). Doubles pipeline is
complete but younger — same architecture, less mature.

## Quickstart (play against the pretrained bot, ~10 min)

You need a Pokémon Showdown server. The bot was developed against a locally
patched server (`pokemon-showdown` fork with offline account registration), but
any server the bot can log into works.

```bash
git clone <this repo> && cd showdown-brain
uv sync

# 1. start a showdown server (see https://github.com/smogon/pokemon-showdown)
#    and register a bot account on it

# 2. point the bot at the server + give it credentials
export SHOWDOWN_BOT_USER=mybot SHOWDOWN_BOT_PASS=hunter2
# (or put {"username": ..., "password": ...} in data/bot-account.json — never committed)

# 3. run it — lobby-visible, accepts singles AND doubles, forever
uv run python scripts/serve.py
```

Then open your server in a browser, find the bot's username, and challenge it
to **[Gen 9] Random Battle** (or Random Doubles, once you flip the format).

Out of the box it loads the pretrained `models/v_model.txt` + `models/pi_model.txt`
shipped in this repo.

## Training your own

```bash
uv sync
uv run python -m sb.harvest --limit 4000 --rating-min 1400   # ladder replays -> data/raw/
uv run python -m sb.parser                                   # -> data/rows_v2.jsonl (full-fidelity states)
uv run python -m sb.parser_doubles                           # -> data/rows_v2_doubles.jsonl
uv run python scripts/train_v_v2.py                          # V(s) over the full state -> models/v2_model.txt
uv run python scripts/train_pi2.py                           # 2-stage policy prior -> data/pi2_*
uv run python scripts/gauntlet.py --games 60                 # plays its past selves + LEARNS from it
```

It learns from every game it plays: finished battles are dumped to
`data/learned_games/`, folded into `rows_personal.jsonl`, and
`models/v2_model.txt` rebuilds on public+personal rows between games
(`sb/learn.py`). See ARCHITECTURE.md for the whole loop.

Splits are by game id, never by row (rows from one game correlate).
LightGBM is capped at 4 threads by default because uncapped training
schedulers will eat your machine.

Watch it learn by playing itself:

```bash
uv run python scripts/selfplay.py    # bot vs poke-env SimpleHeuristicsPlayer
```

## Repo layout

- `src/sb/harvest.py` — replay archive search + log downloader
- `src/sb/parser.py` — showdown protocol → battle state → `(state, action, outcome)` rows (singles)
- `src/sb/parser_doubles.py` — same for random doubles
- `src/sb/features.py` — state dict → numeric vector; `assets/` game data resolution
- `src/sb/transitions.py` — gen-9 damage formula + non-damage move effects, in dict-space
- `src/sb/showdown_state.py` — poke-env live battle → parser-shaped state dict
- `src/sb/agent.py` — VsAgent: one-ply expectimax player
- `src/sb/pi2.py` — 2-stage policy prior inference
- `scripts/` — train_v, train_pi, train_pi2, selfplay, human
- `assets/` — game data (species, moves, type matrix, semantic tags, randbats sets)
- `models/` — pretrained V / π (git-tracked, ~2 MB total)
- `data/` — harvested replays + rows (gitignored, can be GBs)

Game data in `assets/` comes from the community-maintained
[pokemon-data](https://github.com/cyanidepopcorn/pokemon-data) pipeline
(Smogon sets + TypeSafe semantic tags). Override the location with
`POKEMON_DATA=/path/to/dir`.

## Status & known weaknesses

- V(s) AUC ~0.64 — mid-game EVs live in a narrow band, so decisions cluster.
  More (and higher-rated) replays is the main lever.
- One-ply search: no move-sequencing awareness (setup → attack patterns are
  emergent from V, not planned).
- Opponent model assumes a π-weighted damaging response; status/protect play is
  under-modeled.
- Doubles support is young — protect/redirect positioning is not modeled.

## License

MIT (see LICENSE). Replay data is downloaded from the public Showdown replay
archive — respect their rate limits (the harvester sleeps between requests).
