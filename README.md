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

- **V(s)** — LightGBM win predictor over a perspective-symmetric feature vector
  (species, stats, HP, boosts, status, item/ability semantic tags, hazards,
  weather, terrain, tera, …) trained on ~200k decision points from ladder replays.
- **Opponent model** — the opponent's response is scored with a gen-9 damage
  formula (stats, boosts, STAB, type matrix, weather/terrain, accuracy) over the
  moves their species could be running, weighted by the π prior.
- **π(a|s)** — two-stage policy clone (move-vs-switch classifier + per-class
  rankers) learned from what humans actually did in similar states.
- **Terastallization** — evaluated as a first-class candidate whenever it's
  available, with tera typing handled in both offense and defense.

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

# 3. run it — waits for ONE challenge from anyone
uv run python scripts/human.py
```

Then open your server in a browser, find the bot's username, and challenge it
to **[Gen 9] Random Battle** (or Random Doubles, once you flip the format).

Out of the box it loads the pretrained `models/v_model.txt` + `models/pi_model.txt`
shipped in this repo.

## Training your own

```bash
uv sync
uv run python -m sb.harvest --limit 4000 --rating-min 1400   # download ladder replays -> data/raw/
uv run python -m sb.parser                                   # protocol logs -> data/rows.jsonl
uv run python scripts/train_v.py                             # V(s) win predictor -> models/v_model.txt
uv run python scripts/train_pi.py                            # flat pi baseline (legacy)
uv run python scripts/train_pi2.py                           # 2-stage pi -> models/pi2_*.txt
```

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
