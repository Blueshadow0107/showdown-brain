// schema.ts — types for the clean game_data (matches clean.mjs output)
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
