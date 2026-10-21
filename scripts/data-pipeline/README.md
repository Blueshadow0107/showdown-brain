# data-pipeline

How the game data in `assets/` was built. You don't need to run this to use the
bot — `assets/` ships the finished files. It's here so the data is auditable.

Scripts read and write next to themselves, so run them from a scratch copy of
this folder.

1. **`clean.mjs`** — Pokémon Showdown data tables (`abilities.ts`, `items.ts`,
   `moves.ts`, `pokedex.ts`, `text-*.ts`, `random-sets.ts`) plus the randbats /
   randdoubles set JSON -> clean JSON (`species.json`, `moves.json`,
   `abilities.json`, `items.json`, `typechart.json`, `randbats-sets.json`,
   `randdoubles-sets.json`, `game_data.json`) and `schema.ts`. Inputs go in a
   `raw/` folder next to the script, copied from
   [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown)
   (`data/`). Needs a Node with native TypeScript stripping.
2. **`tag.mjs`** — rule-based tags per ability/item/move -> `tags.json`. Entries it
   can't classify by rule are flagged `needsSemantic`.
3. **`tag_jev.py`** — semantic pass over the flagged entries via the TypeSafe
   System One API -> `tags_semantic.json` (resumable). Needs `TYPESAFE_API_KEY`;
   try `--limit 8` first.

`assets/tags_final.json` is the merge of rule tags and semantic judgments. That
merge was done by hand and is **not scripted here**.
