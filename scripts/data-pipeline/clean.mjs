// clean.mjs — showdown TS data tables -> clean JSON (+ schema.ts)
// run: node clean.mjs
import fs from 'node:fs';

const RAW = new URL('./raw/', import.meta.url).pathname;
const OUT = new URL('./', import.meta.url).pathname;

/* ---- load a PS data table via node's native TS stripping ---- */
async function loadTS(file, exportName) {
  const mod = await import(RAW + file);
  const t = mod[exportName];
  if (!t) throw new Error(`${file}: export ${exportName} not found (have: ${Object.keys(mod)})`);
  return t;
}

/* ---- deep-clean: functions -> markers (counted), drop undefined ---- */
let dropped = {};
function deepClean(v, path, file) {
  if (typeof v === 'function') {
    const key = `${file}:${path.split('.').pop()}`;
    dropped[key] = (dropped[key] || 0) + 1;
    return '__dynamic__';
  }
  if (Array.isArray(v)) return v.map((x, i) => deepClean(x, `${path}[${i}]`, file));
  if (v && typeof v === 'object') {
    const o = {};
    for (const k of Object.keys(v).sort()) {
      const c = deepClean(v[k], path ? `${path}.${k}` : k, file);
      if (c !== undefined) o[k] = c;
    }
    return o;
  }
  return v;
}

/* ================= moves ================= */
const movesRaw = await loadTS('moves.ts', 'Moves');
const moves = {};
for (const [id, m] of Object.entries(movesRaw)) {
  const o = {};
  o.name = m.name;
  o.num = m.num;
  o.type = m.type;
  o.category = (m.category || 'Status').toLowerCase();
  o.power = m.basePower ?? null;
  if (m.basePowerCallback) o.dynamicPower = true;
  o.accuracy = m.accuracy === true ? null : (m.accuracy ?? null); // null = never misses
  o.pp = m.pp ?? null;
  o.priority = m.priority || 0;
  o.target = m.target || 'normal';
  if (m.flags) o.flags = Object.keys(m.flags).sort();
  // effect fields — keep whatever exists (all plain data)
  for (const k of ['boosts','self','secondary','secondaries','heal','status','volatileStatus',
    'weather','terrain','sideCondition','slotCondition','forceSwitch','selfSwitch',
    'drain','recoil','ohko','damage','condition','stealsBoosts','breaksProtect',
    'hasCrashDamage','ignoresImmunity','mindBlownRecoil','struggleRecoil','noPPBoosts',
    'isNonstandard','pressureTarget','maxMove','zMove','gmaxPower']) {
    if (m[k] !== undefined) o[k] = deepClean(m[k], k, 'moves');
  }
  moves[id] = o;
}

/* ================= abilities ================= */
const abilitiesRaw = await loadTS('abilities.ts', 'Abilities');
const abilities = {};
for (const [id, a] of Object.entries(abilitiesRaw)) {
  const c = deepClean(a, '', 'abilities');
  if (JSON.stringify(c).includes('__dynamic__')) c.hasScripts = true;
  abilities[id] = c;
}

/* ================= items ================= */
const itemsRaw = await loadTS('items.ts', 'Items');
const items = {};
for (const [id, it] of Object.entries(itemsRaw)) {
  const c = deepClean(it, '', 'items');
  // flag entries whose effect logic lives in code (functions we replaced)
  if (JSON.stringify(c).includes('__dynamic__')) c.hasScripts = true;
  items[id] = c;
}

/* ================= species (pokedex) ================= */
const dexRaw = await loadTS('pokedex.ts', 'Pokedex');
const KEEP = ['name','num','types','baseStats','abilities','weightkg','heightm',
  'baseSpecies','forme','otherFormes','cosmeticFormes','evos','prevo',
  'isNonstandard','gender','requiredItems','changesFrom'];
const species = {};
for (const [id, p] of Object.entries(dexRaw)) {
  const o = {};
  for (const k of KEEP) if (p[k] !== undefined) o[k] = p[k];
  species[id] = o;
}

/* ================= type chart -> dense multiplier matrix ================= */
// PS encoding in damageTaken (defender's view): 0 neutral, 1 weak(x2), 2 resist(x0.5), 3 immune(x0)
// PS keys are lowercase ids; damageTaken keys are canonical ("Grass"). Canonical wins.
const chartRaw = await loadTS('typechart.ts', 'TypeChart');
const DECODE = {0: 1, 1: 2, 2: 0.5, 3: 0};
const canon = t => t.charAt(0).toUpperCase() + t.slice(1);
const allTypes = Object.keys(chartRaw).map(canon).sort(); // 18 + Stellar
const matrix = {};
for (const defRaw of Object.keys(chartRaw)) {
  const dt = chartRaw[defRaw].damageTaken;
  const def = canon(defRaw);
  for (const attRaw of Object.keys(dt)) {
    const att = canon(attRaw);
    (matrix[att] ||= {})[def] = DECODE[dt[attRaw]];
  }
}
// fill missing pairs as neutral (e.g. Stellar pairs PS leaves blank)
for (const att of allTypes) for (const def of allTypes) {
  (matrix[att] ||= {})[def] ??= 1;
}

/* ---- cross-validate against the hand-built chart in pokemon-systems.html ---- */
const PRIMER = {
  "Normal":{"Rock":0.5,"Ghost":0,"Steel":0.5},"Fire":{"Fire":0.5,"Water":0.5,"Grass":2,"Ice":2,"Bug":2,"Rock":0.5,"Dragon":0.5,"Steel":2},
  "Water":{"Fire":2,"Water":0.5,"Grass":0.5,"Ground":2,"Rock":2,"Dragon":0.5},"Electric":{"Water":2,"Electric":0.5,"Grass":0.5,"Ground":0,"Flying":2,"Dragon":0.5},
  "Grass":{"Fire":0.5,"Water":2,"Grass":0.5,"Poison":0.5,"Ground":2,"Flying":0.5,"Bug":0.5,"Rock":2,"Dragon":0.5,"Steel":0.5},
  "Ice":{"Fire":0.5,"Water":0.5,"Grass":2,"Ice":0.5,"Ground":2,"Flying":2,"Dragon":2,"Steel":0.5},
  "Fighting":{"Normal":2,"Ice":2,"Poison":0.5,"Flying":0.5,"Psychic":0.5,"Bug":0.5,"Rock":2,"Ghost":0,"Dark":2,"Steel":2,"Fairy":0.5},
  "Poison":{"Grass":2,"Poison":0.5,"Ground":0.5,"Rock":0.5,"Ghost":0.5,"Steel":0,"Fairy":2},
  "Ground":{"Fire":2,"Electric":2,"Grass":0.5,"Poison":2,"Flying":0,"Bug":0.5,"Rock":2,"Steel":2},
  "Flying":{"Grass":2,"Electric":0.5,"Fighting":2,"Bug":2,"Rock":0.5,"Steel":0.5},
  "Psychic":{"Fighting":2,"Poison":2,"Psychic":0.5,"Dark":0,"Steel":0.5},
  "Bug":{"Grass":2,"Fire":0.5,"Fighting":0.5,"Poison":0.5,"Flying":0.5,"Psychic":2,"Ghost":0.5,"Dark":2,"Steel":0.5,"Fairy":0.5},
  "Rock":{"Fire":2,"Ice":2,"Fighting":0.5,"Ground":0.5,"Flying":2,"Bug":2,"Steel":0.5},
  "Ghost":{"Normal":0,"Psychic":2,"Ghost":2,"Dark":0.5},"Dragon":{"Dragon":2,"Steel":0.5,"Fairy":0},
  "Dark":{"Fighting":0.5,"Psychic":2,"Ghost":2,"Dark":0.5,"Fairy":0.5},
  "Steel":{"Fire":0.5,"Water":0.5,"Electric":0.5,"Ice":2,"Rock":2,"Steel":0.5,"Fairy":2},
  "Fairy":{"Fire":0.5,"Fighting":2,"Poison":0.5,"Dragon":2,"Dark":2,"Steel":0.5}
};
const core18 = allTypes.filter(t => t !== 'Stellar');
let mismatches = 0;
for (const att of Object.keys(PRIMER)) for (const def of Object.keys(PRIMER[att])) {
  const official = matrix[att]?.[def];
  const mine = PRIMER[att][def];
  if (official !== mine) { mismatches++; console.log(`MISMATCH ${att}->${def}: primer=${mine} official=${official}`); }
}
console.log(`typechart cross-check: ${mismatches} mismatches vs hand-built primer`);

/* ---- merge flavour/effect text (data/text/*.ts) into the three tables ---- */
const TEXT = {
  moves: await loadTS('text-moves.ts', 'MovesText'),
  abilities: await loadTS('text-abilities.ts', 'AbilitiesText'),
  items: await loadTS('text-items.ts', 'ItemsText'),
};
const TABLES = {moves, abilities, items};
for (const [kind, table] of Object.entries(TEXT)) {
  for (const [id, txt] of Object.entries(table)) {
    const e = TABLES[kind][id];
    if (!e) continue;
    if (txt.shortDesc) e.shortDesc = txt.shortDesc;
    if (txt.desc) e.desc = txt.desc;
  }
}

/* ================= write files ================= */
const write = (name, obj) => {
  fs.writeFileSync(OUT + name, JSON.stringify(obj, null, 1));
  return fs.statSync(OUT + name).size;
};
const randbats = JSON.parse(fs.readFileSync(RAW + 'randbats-sets.json', 'utf8'));
const randdoubles = JSON.parse(fs.readFileSync(RAW + 'randdoubles-sets.json', 'utf8'));

const sizes = {
  'moves.json': write('moves.json', moves),
  'abilities.json': write('abilities.json', abilities),
  'items.json': write('items.json', items),
  'species.json': write('species.json', species),
  'typechart.json': write('typechart.json', {types: allTypes, matrix}),
  'randbats-sets.json': write('randbats-sets.json', randbats),
  'randdoubles-sets.json': write('randdoubles-sets.json', randdoubles),
};
const gameData = {
  meta: {source: 'smogon/pokemon-showdown master (gen 9)', built: new Date().toISOString().slice(0,10)},
  types: allTypes,
  matrix,
  moves, abilities, items, species,
  randomBattles: {singles: randbats, doubles: randdoubles},
};
sizes['game_data.json'] = write('game_data.json', gameData);

/* ================= schema.ts ================= */
fs.writeFileSync(OUT + 'schema.ts', `// schema.ts — types for the clean game_data (matches clean.mjs output)
export type TypeName = string; // see types[]
export type MoveCategory = 'physical' | 'special' | 'status';
export type Target = 'normal' | 'self' | 'allySide' | 'foeSide' | 'all' | 'allAdjacent'
  | 'allAdjacentFoes' | 'allyTeam' | 'randomNormal' | 'scripted' | 'adjacentAlly' | 'adjacentFoe';

export interface Move {
  name: string;
  num: number;
  type: TypeName;
  category: MoveCategory;
  power: number | null;        // null = no fixed power (status / damage-fields)
  dynamicPower?: true;         // power computed by script (weight, hp%, etc.)
  accuracy: number | null;     // null = never misses
  pp: number | null;
  priority: number;            // negative = move last (Trick Room handled separately)
  target: Target;
  flags?: string[];            // contact, protect, mirror, punch, bite, sound, bullet, pulse, slicing, wind, powder, heal, defrost, ...
  boosts?: Record<string, number>;           // stat changes on target, e.g. {atk: -1}
  self?: object;               // self-targeted effects (volatileStatus, boosts, boosts2...)
  secondary?: object;          // single secondary effect {chance, status, boosts, volatileStatus}
  secondaries?: object[];
  heal?: number[];             // [numerator, denominator] of max HP
  status?: string;             // brn | par | psn | tox | slp | frz
  volatileStatus?: string;     // confusion, trapped, substitute, etc.
  weather?: string;            // raindance | sunnyday | sandstorm | snowscape
  terrain?: string;            // electricterrain | grassyterrain | mistyterrain | psychicterrain
  sideCondition?: string;      // stealthrock | spikes | toxicspikes | stickyweb | tailwind | reflect | lightscreen | auroraveil
  slotCondition?: string;
  forceSwitch?: boolean | number;  // phazing / red-card style
  selfSwitch?: string;         // 'copyvolatile' | 'shedtail' — pivot moves
  drain?: number[];            // [num, den] of damage dealt
  recoil?: number[];           // [num, den] of damage dealt
  ohko?: boolean | string;     // sheer cold etc.
  damage?: number | 'level' | 'half'; // fixed damage
  stealsBoosts?: boolean;
  breaksProtect?: boolean;
  isNonstandard?: string;      // 'Past' | 'Unobtainable' | 'Gigantamax' | ...
  hasScripts?: never;
  [k: string]: unknown;        // forward-compat: extra data fields preserved
}
export interface Ability { name: string; num: number; rating: number; desc?: string; shortDesc?: string; isNonstandard?: string; hasScripts?: true; [k: string]: unknown; }
export interface Item { name: string; num: number; gen: number; desc?: string; shortDesc?: string; isChoice?: boolean; isBerry?: boolean; megaStone?: string; fling?: {basePower: number; status?: string; volatileStatus?: string}; naturalGift?: {basePower: number; type: TypeName}; hasScripts?: true; [k: string]: unknown; }
export interface Species {
  name: string; num: number; types: TypeName[];
  baseStats: {hp: number; atk: number; def: number; spa: number; spd: number; spe: number};
  abilities: {'0': string; '1'?: string; H?: string; S?: string};
  weightkg: number; heightm: number;
  baseSpecies?: string; forme?: string; otherFormes?: string[]; cosmeticFormes?: string[];
  evos?: string[]; prevo?: string; isNonstandard?: string; [k: string]: unknown;
}
export interface RandbatsSet { role: string; movepool: string[]; abilities: string[]; teraTypes?: string[]; evs?: object; ivs?: object; nature?: string }
export interface GameData {
  meta: {source: string; built: string};
  types: TypeName[];
  matrix: Record<TypeName, Record<TypeName, number>>;  // matrix[attacker][defender] = multiplier
  moves: Record<string, Move>;
  abilities: Record<string, Ability>;
  items: Record<string, Item>;
  species: Record<string, Species>;
  randomBattles: {singles: Record<string, {level: number; sets: RandbatsSet[]}>, doubles: Record<string, {level: number; sets: RandbatsSet[]}>};
}
`);
sizes['schema.ts'] = fs.statSync(OUT + 'schema.ts').size;

/* ================= report ================= */
const countIf = (o, f) => Object.values(o).filter(f).length;
console.log('\n=== game-data build report ===');
for (const [k, v] of Object.entries(sizes)) console.log(`  ${k.padEnd(24)} ${(v/1024).toFixed(0).padStart(5)} KB`);
console.log(`  moves:      ${Object.keys(moves).length}  (${countIf(moves, m=>m.dynamicPower)} dynamic-power)`);
console.log(`  abilities:  ${Object.keys(abilities).length}  (${countIf(abilities, a=>a.hasScripts)} w/ scripts)`);
console.log(`  items:      ${Object.keys(items).length}  (${countIf(items, i=>i.hasScripts)} w/ scripts)`);
console.log(`  species:    ${Object.keys(species).length}`);
console.log(`  types:      ${allTypes.length} (${allTypes.join(', ')})`);
console.log(`  randbats sets: singles ${Object.keys(randbats).length} species, doubles ${Object.keys(randdoubles).length} species`);
console.log('\n  dropped functions by field:');
for (const [k, v] of Object.entries(dropped).sort((a,b)=>b[1]-a[1])) console.log(`    ${k}  x${v}`);
