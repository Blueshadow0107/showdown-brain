// tag.mjs — rules-first semantic tagging over game_data.json
// emits tags.json in the SAME per-entity schema a TypeSafe Choice/Noul/Score
// pass would produce — when you get an API key, regenerate with the semantic
// pass and the downstream Markov code won't know the difference.
// run: node tag.mjs
import fs from 'node:fs';
const OUT = new URL('./', import.meta.url).pathname;
const d = JSON.parse(fs.readFileSync(OUT + 'game_data.json', 'utf8'));

const cap = x => Math.max(0, Math.min(1, x));
const has = (e, ...ks) => ks.some(k => e[k] !== undefined);
const NA = () => ({});

/* ============ MOVES ============ */
const HAZARDS = new Set(['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']);
const SCREENS = new Set(['reflect', 'lightscreen', 'auroraveil']);
const SPINNERS = new Set(['rapidspin', 'defog', 'mortalspin', 'courtchange', 'tidyup', 'crestedcleansing?']);
const PROTECTS = new Set(['protect', 'detect', 'kingsshield', 'spikyshield', 'banefulbunker', 'silktrap', 'obstruct', 'burningbulwark', 'maxguard']);
const WISHES = new Set(['wish', 'healingwish', 'lunardance', 'junglehealing', 'lifedew', ' Floralhealing'.toLowerCase()]);

function tagMove(m) {
  const tags = new Set(), scores = {aggression: 0, setup: 0, control: 0, tempo: 0, support: 0};
  const id = m.name.toLowerCase().replace(/[^a-z0-9]/g, '');
  const sec = m.secondary || (m.secondaries && m.secondaries[0]);
  const selfBoost = m.boosts && m.target === 'self'
    ? Object.values(m.boosts).reduce((a, b) => a + Math.max(0, b), 0) : 0;
  const secSelfBoost = (sec && sec.self && sec.self.boosts)
    ? Object.values(sec.self.boosts).reduce((a, b) => a + Math.max(0, b), 0) : 0;
  const drop = m.boosts ? Object.values(m.boosts).some(v => v < 0) : (sec && sec.boosts ? Object.values(sec.boosts).some(v => v < 0) : false);
  const inflictChance = m.status ? 1 : (sec && sec.status ? (sec.chance ?? 30) / 100 : 0);

  // damage
  if (m.category !== 'status' && (m.power > 0 || m.dynamicPower)) {
    tags.add('damage');
    scores.aggression = cap((m.power || 80) / 150) * ((m.accuracy ?? 100) / 100) * (m.dynamicPower ? 0.9 : 1);
  }
  if (m.category !== 'status' && !m.power && !m.dynamicPower) { tags.add('fixed-damage'); scores.aggression = 0.4; }

  // setup
  const setupAmt = Math.max(selfBoost, secSelfBoost);
  if (setupAmt > 0) { tags.add('setup'); scores.setup = cap(setupAmt / 4); }
  if (drop) { tags.add('debuff'); scores.control = Math.max(scores.control, 0.45); }

  // pivot / phazing / protect
  if (m.selfSwitch) { tags.add('pivot'); scores.tempo = Math.max(scores.tempo, 0.8); }
  if (m.forceSwitch) { tags.add('phazing'); scores.control = Math.max(scores.control, 0.7); }
  if (PROTECTS.has(id) || m.volatileStatus === 'protect') { tags.add('protect'); scores.tempo = Math.max(scores.tempo, 0.65); scores.control = Math.max(scores.control, 0.4); }

  // hazards / screens / removal
  if (HAZARDS.has(m.sideCondition)) { tags.add('hazard'); scores.control = Math.max(scores.control, 0.85); }
  if (SCREENS.has(m.sideCondition)) { tags.add('screen'); scores.support = Math.max(scores.support, 0.6); scores.control = Math.max(scores.control, 0.5); }
  if (SPINNERS.has(id)) { tags.add('hazard-removal'); scores.support = Math.max(scores.support, 0.8); }

  // field setters / speed control
  if (m.weather) { tags.add('weather-set'); scores.tempo = Math.max(scores.tempo, 0.5); scores.control = Math.max(scores.control, 0.4); }
  if (m.terrain) { tags.add('terrain-set'); scores.tempo = Math.max(scores.tempo, 0.5); scores.control = Math.max(scores.control, 0.4); }
  const SPEED_MOVES = new Set(['tailwind', 'trickroom', 'thunderwave', 'stringshot', 'cottonspore', 'scaryface', 'stickyweb', 'icywind', 'electroweb', 'rocktomb', ' Bulldoze'.toLowerCase()]);
  if (m.priority > 0) { tags.add('priority'); scores.tempo = Math.max(scores.tempo, cap(m.priority / 4)); }
  if (SPEED_MOVES.has(id) || m.sideCondition === 'tailwind' || m.status === 'par') { tags.add('speed-control'); scores.control = Math.max(scores.control, 0.6); }

  // status / recovery / drain / recoil / ohko / spread
  if (inflictChance > 0) { tags.add('status-inflict'); scores.control = Math.max(scores.control, 0.55 * inflictChance); }
  if (m.heal || WISHES.has(id)) { tags.add('recovery'); scores.support = Math.max(scores.support, m.heal ? m.heal[0] / m.heal[1] : 0.9); }
  if (m.drain) { tags.add('drain'); scores.aggression = Math.max(scores.aggression, 0.55); }
  if (has(m, 'recoil', 'mindBlownRecoil', 'hasCrashDamage', 'struggleRecoil')) tags.add('recoil');
  if (m.ohko) { tags.add('ohko'); scores.aggression = 0.3; }
  if (['all', 'allAdjacent', 'allAdjacentFoes'].includes(m.target)) { tags.add('spread'); scores.aggression *= 1.2; scores.aggression = cap(scores.aggression); }
  if (m.stealsBoosts) { tags.add('anti-setup'); scores.control = Math.max(scores.control, 0.6); }
  if (m.breaksProtect) { tags.add('anti-protect'); scores.tempo = Math.max(scores.tempo, 0.5); }
  if (m.volatileStatus === 'substitute') { tags.add('substitute'); scores.control = Math.max(scores.control, 0.5); }

  const hitCount = tags.size;
  return {
    tags: [...tags].sort(),
    scores: Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, +v.toFixed(3)])),
    confidence: 'rule',
    needsSemantic: hitCount === 0 && m.isNonstandard !== 'Past' && m.category === 'status',
  };
}

/* ============ ABILITIES ============ */
function tagAbility(a) {
  const tags = new Set(), scores = {offense: 0, defense: 0, speed: 0, momentum: 0, field: 0};
  const t = (a.shortDesc || a.desc || '').toLowerCase(); // shortDesc only — long descs mention effects on OTHER mons
  const id = a.name.toLowerCase().replace(/[^a-z0-9]/g, '');
  const R = (re) => re.test(t);

  if (R(/summons?( a)? (rain|harsh sunlight|sun|sandstorm|snow)|weather/i) && !R(/immune|halves|reduces/i)) { tags.add('weather'); scores.field = 0.9; }
  if (R(/terrain/i)) { tags.add('terrain'); scores.field = 0.9; }
  if (R(/speed|swift|chlorophyll|rush|unburden/i) && R(/speed stat|doubles? its speed|raised|boost|swift swim|chlorophyll|rush|unburden|twice/i)) { tags.add('speed'); scores.speed = 0.85; }
  if (!tags.has('immunity') && R(/attack stat|special attack stat|huge power|pure power|technician|sheer force|strong jaw|iron fist|tough claws|berserk|anger point|guts/i) && !R(/lowers|reduces/i)) { tags.add('offense'); scores.offense = 0.8; }
  if (R(/defense stat|special defense stat|reduces? (?:incoming )?damage|takes? (?:less|reduced)|filter|solid rock|thick fat|fur coat|multiscale|sturdy/i)) { tags.add('defense'); scores.defense = 0.8; }
  if (R(/immune|absorb|grants immunity|cannot be (?:hit|struck)|evades?/i)) { tags.add('immunity'); scores.defense = Math.max(scores.defense, 0.9); }
  if (R(/restores?|heals?|recover|regenerat|rain dish|poison heal|ice body|natural cure/i)) { tags.add('recovery'); scores.momentum = Math.max(scores.momentum, 0.75); }
  if (R(/switch(?:es)? out|when it switches|baton pass|teleport|flee/i)) { tags.add('switching'); scores.momentum = Math.max(scores.momentum, 0.8); }
  if (R(/on switch-in|intimidate|download|trace|frisk|anticipation/i)) { tags.add('entry'); scores.momentum = Math.max(scores.momentum, 0.5); }
  if (R(/contact/i)) { tags.add('contact'); scores.defense = Math.max(scores.defense, 0.5); }
  if (R(/hazard|stealth rock|spikes|sticky web/i)) { tags.add('hazard-interact'); scores.momentum = Math.max(scores.momentum, 0.6); }
  if (R(/status|burn|poison|paralyz|sleep/i) && !R(/immune|heals?|restores?/i)) { tags.add('status'); scores.offense = Math.max(scores.offense, 0.5); }
  if (R(/critical|crit/i)) { tags.add('crit'); scores.offense = Math.max(scores.offense, 0.6); }
  if (R(/priority|prankster|triage/i)) { tags.add('priority'); scores.speed = Math.max(scores.speed, 0.7); }
  if (R(/boost|strengthens|raises|moxie|beast boost|grim neigh|chilling neigh|defiant|competitive/i)) { tags.add('boost'); scores.offense = Math.max(scores.offense, 0.7); }

  scores.impact = +(cap(a.rating / 5)).toFixed(3); // showdown's own 0-5 rating, normalised
  return {
    tags: [...tags].sort(),
    scores: Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, +v.toFixed(3)])),
    confidence: 'rule',
    needsSemantic: tags.size <= 1 && a.hasScripts === true,
  };
}

/* ============ ITEMS ============ */
function tagItem(it) {
  const tags = new Set(), scores = {offense: 0, defense: 0, speed: 0, utility: 0};
  const t = (it.shortDesc || it.desc || '').toLowerCase();
  const R = (re) => re.test(t);
  const mult = () => { const m = t.match(/([12](?:\.\d+)?)\s*(?:x|×|times)/) || t.match(/(?:x|×)\s*([12](?:\.\d+)?)/); return m ? +m[1] : null; };

  if (it.isChoice) { tags.add('choice'); scores.offense = 0.85; scores.speed = R(/speed/) ? 0.85 : 0; }
  if (it.isBerry) { tags.add('berry'); scores.utility = Math.max(scores.utility, 0.5); }
  if (R(/attack|special attack|damage|power/i) && !R(/defense|hp/i)) { tags.add('offense'); scores.offense = Math.max(scores.offense, mult() ? cap((mult() - 1) / 0.5 * 0.5 + 0.5) : 0.65); }
  if (R(/speed/i)) { tags.add('speed'); scores.speed = Math.max(scores.speed, mult() ? cap((mult() - 1) / 0.5 * 0.4 + 0.5) : 0.6); }
  if (R(/restores?|heals?|recover|leftovers|sitrus|shell bell/i)) { tags.add('recovery'); scores.defense = Math.max(scores.defense, 0.75); }
  if (R(/defense stat|special defense stat|reduces? (?:incoming )?damage|takes? (?:less|reduced)|survives? (?:an? )?(?:hit|attack)|would ko|filter|solid rock|thick fat|fur coat|multiscale|sturdy/i)) { tags.add('defense'); scores.defense = Math.max(scores.defense, mult() ? cap((mult() - 1) / 0.5 * 0.3 + 0.6) : 0.7); }
  if (R(/hazard|stealth rock|spikes|web/i)) { tags.add('hazard-interact'); scores.utility = Math.max(scores.utility, 0.7); }
  if (R(/switch|eject|red card|shed shell|escape/i)) { tags.add('switching'); scores.utility = Math.max(scores.utility, 0.8); }
  if (R(/weather|rain|sun|sand|snow/i)) { tags.add('weather'); scores.utility = Math.max(scores.utility, 0.7); }
  if (R(/terrain/i)) { tags.add('terrain'); scores.utility = Math.max(scores.utility, 0.7); }
  if (R(/screen|reflect|light screen|aurora/i)) { tags.add('screen'); scores.utility = Math.max(scores.utility, 0.7); }
  if (R(/status|burn|poison|paralyz|sleep|lum/i)) { tags.add('status'); scores.utility = Math.max(scores.utility, 0.6); }
  if (R(/priority/i)) { tags.add('priority'); scores.speed = Math.max(scores.speed, 0.7); }
  if (it.megaStone || R(/mega evolve|terastalliz/i)) { tags.add('forme-change'); scores.utility = 1; }
  if (R(/critical|crit/i)) { tags.add('crit'); scores.offense = Math.max(scores.offense, 0.55); }

  return {
    tags: [...tags].sort(),
    scores: Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, +v.toFixed(3)])),
    confidence: 'rule',
    needsSemantic: tags.size <= 1 && it.hasScripts === true,
  };
}

/* ============ run ============ */
const out = {
  meta: {
    method: 'rules-v1',
    note: 'swap for a TypeSafe Choice/Noul/Score pass later — per-entity schema is identical',
    built: new Date().toISOString().slice(0, 10),
  },
  moves: Object.fromEntries(Object.entries(d.moves).map(([k, v]) => [k, tagMove(v)])),
  abilities: Object.fromEntries(Object.entries(d.abilities).map(([k, v]) => [k, tagAbility(v)])),
  items: Object.fromEntries(Object.entries(d.items).map(([k, v]) => [k, tagItem(v)])),
};
fs.writeFileSync(OUT + 'tags.json', JSON.stringify(out, null, 1));

const cov = (o) => {
  const v = Object.values(o);
  return {
    total: v.length,
    tagged: v.filter(x => x.tags.length > 0).length,
    semantic: v.filter(x => x.needsSemantic).length,
  };
};
console.log('=== tag coverage ===');
for (const k of ['moves', 'abilities', 'items']) {
  const c = cov(out[k]);
  console.log(`  ${k.padEnd(10)} ${c.tagged}/${c.total} tagged  ·  ${c.semantic} flagged for TypeSafe pass`);
}
// spot checks
console.log('\n=== spot checks ===');
for (const id of ['uturn', 'stealthrock', 'swordsdance', 'protect', 'thunderwave']) console.log(' move ' + id.padEnd(14), JSON.stringify(out.moves[id].tags), JSON.stringify(out.moves[id].scores));
for (const id of ['regenerator', 'intimidate', 'drizzle', 'levitate']) console.log(' abil ' + id.padEnd(14), JSON.stringify(out.abilities[id]?.tags));
for (const id of ['choicescarf', 'leftovers', 'focussash', 'heavydutyboots']) console.log(' item ' + id.padEnd(14), JSON.stringify(out.items[id]?.tags), JSON.stringify(out.items[id]?.scores));
