# Architecture

One page. If it disagrees with the code, the code is wrong — fix the code.

## The loop

```
ladder replays                    your games + gauntlet games          FFA: public harvest + self-play + your games
      |                                    |                                    |
  sb.harvest                       sb.learn.dump_battle                 sb.harvest --format gen9freeforallrandombattle
      |                                    |                             scripts/ffa_selfplay.py (4 accounts)
  data/raw/*.log                   data/learned_games/*.log                   data/{raw,learned_games}/*.log
      |                                    |                                    |
      +------------ sb.parser ------------+        (gen9randombattle -> rows_v2.jsonl)
      +---------- sb.parser_doubles ------+        (doubles        -> rows_v2_doubles.jsonl)
      +---------- sb.parser_ffa --------- +        (freeforall     -> rows_ffa*.jsonl, pairwise rows)
                         |
              features.py  (v1, 300 dims, momentum)
              features_v2.py (v2, 393 dims, full-fidelity)
                         |
        train_v.py / train_v_v2.py / train_v_ffa.py / train_pi2*.py   -> models/
                         |
              VsAgent (singles)  VsAgentDoubles  VsAgentFFA   <- sb/heuristics.py = all tuning constants
                         |
              scripts/serve.py (the lobby bot, accepts all three formats)
              scripts/gauntlet.py (plays its past selves, feeds learn.py)
              scripts/ffa_selfplay.py (4-seat FFA data generation)
```

## Board states (the canonical currency)

`parser.py` and `parser_doubles.py` turn protocol logs into **rows**:
`{game, turn, player, [slot], action_kind, action_id, state, state_pre, outcome}`.

- `state` = post-decision snapshot (V trains on this: value of positions
  reached after actions — exactly what the search's leaf states look like)
- `state_pre` = pre-decision snapshot (π trains on this; leak-checked)
- **state v2** is a strict superset of v1: `turn`, side-level `screens`
  (turns remaining), `tera_used`, `last_action`, and `team[]` — every known
  team member with hp/status/boosts/pp_used/last_move/consecutive/
  times_entered/knowledge flags. Silent mechanics (Regenerator ⅓ heal,
  Natural Cure on switch-out) are mirrored when the ability is known.
- v1 keys never changed shape, so the v1 featurizer still runs on v2 rows.

## The player

`VsAgent.choose_move` = one-ply expectimax per legal action:

```
EV(a) = V( clone(state) |> apply a |> apply entry hazards |> opponent response )
opponent response = pi2-weighted expected damage  x  P(move | their view)
                        + your own hazard/screen/status rules from transitions.py
```

then four judgment calls from `sb/heuristics.py` (the ONLY magic numbers):
switch-commit threshold, repetition tax, last-mon urgency, tera margin.
Unmodelable move effects take the unmodelled penalty.

`transitions.py` is the gen-9 physics: damage (stats/boosts/STAB/matrix/
weather/terrain/screens/accuracy/ordering), kill-clocks, effective speed,
entry hazards, forme resolution. Everything physical lives there; nothing
learned lives there.

## Brains

A brain is `{model, featurizer, pi2}` — see `VsAgent(brain=...)`.
`models/v2_model.txt` + featurizer `v2` is the current build.
`models/history/v1_classic.txt` + featurizer `v1-classic` (momentum stripped)
is the dim-287 ancestor for gauntlets.

## FFA (free-for-all)

4 players (`gen9freeforallrandombattle`), 6 mons each. One design decision
drives everything: **pairwise projection**. At any decision the 4-player game
becomes one canonical 2-key v2 state per live opponent (`my` vs that foe,
via `showdown_state_ffa.battle_to_state_ffa`), so the 393-dim featurizer and
the V models work unchanged.

- **Rows** — `parser_ffa` emits one row per (decision × live opponent); the
  only schema addition is `foe_n` (projected side). `outcome` = 1 iff the
  acting player wins the whole FFA (placements 2nd–4th are 0). Same keys
  otherwise — the v2 featurizer can't tell FFA rows from ladder rows.
- **Brains** — `VsAgentFFA`: EV(a) = Σ_i w_i·V(next vs foe_i) −
  `ffa_retaliation_weight`·Σ_j retaliation_j + `ffa_ko_bonus`·KOs +
  `ffa_focus_bonus`·(1−hp)·hits. w_i piles onto the strongest board via
  `ffa_threat_weight`; retaliation is pi2-weighted per foe (the 1v1 prior,
  used as-is). `models/v_ffa.txt` when present, else zero-shot v2. Foe
  targeting: `create_order(move, move_target=loc)` with `sb.ffa.target_loc`.
- **Live bridge** — poke-env is 2-player; `FFABattle` + `install()`
  (sb/ffa_poke_env.py) fix title/player events and bucket mons per side.
  Eliminated sides send no decisions; `|win|` is the only end signal.
- **Sets** — transitions loads `SETS_BY_KIND = {singles, ffa}` from vendored
  assets; `T.set_kind("ffa")` per decision (VsAgentFFA does this; the default
  stays "singles").
- **Physics deltas** — spread moves ×0.5 (`sb.ffa.SPREAD_MODIFIER`); foeSide
  hazards hit all foes; a single-target move whose target fainted first does
  NOT retarget (whiffs) — vulture KOs are real.

## Learning from its own games

`serve.py`/`gauntlet.py` dump every finished battle's protocol log to
`data/learned_games/`. `sb.learn.learn_new_games()` parses new logs, appends
to `data/rows_personal.jsonl`, and rebuilds `v2_model.txt` on public+personal
rows (val split stays public-only). Called between games by the gauntlet;
call it after sessions by hand: `uv run python -c "from sb.learn import
learn_new_games as l; l()"` from the project root.

## Rules for contributors (and future agents)

1. Physics in transitions.py. Judgment constants in heuristics.py. Nothing
   magic anywhere else.
2. New state fields: add to BOTH parsers and BOTH live mappers, keep v1
   keys intact, document in this file. (FFA is exempt: pairwise projected
   states are built only by showdown_state_ffa and never touch the
   2-player parsers.)
3. Models are trained artifacts — never hand-edit; retrain via train_*.py.
4. rows_v2*.jsonl is the training corpus; rows*.jsonl (v1) is frozen history.
   FFA pairwise rows live in rows_ffa{,_personal}.jsonl.
5. Train/eval splits are by game id; the public split never includes
   personal games.
