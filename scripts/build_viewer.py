"""build_viewer.py — emit a self-contained HTML viewer + a JSON Crack shape file
for assets/gamebundle.json.

  assets/gamebundle.html   lazy-rendered tree + search + decoded rows + typechart
                           (open in any browser; no node limits, no deps)
  assets/gamebundle.shape.json   structure-only summary: counts, vocabs, and the
                           first few rows of each table — small enough for
                           JSON Crack's ~1500-node cap
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "assets" / "gamebundle.json"
OUT_HTML = ROOT / "assets" / "gamebundle.html"
OUT_SHAPE = ROOT / "assets" / "gamebundle.shape.json"

SAMPLE = 3  # rows to inline in the shape file


def main():
    b = json.load(open(BUNDLE))
    raw = json.dumps(b, separators=(",", ":"))

    # ---- shape file (jsoncrack-friendly) -----------------------------------
    shape = {"_schema": b["_schema"], "vocab": b["vocab"],
             "counts": {}, "samples": {}}
    for name in ("moves", "abilities", "items", "species"):
        t = b[name]
        shape["counts"][name] = len(t["ids"])
        key = "tags" if name in ("abilities", "items") else "rows"
        shape["samples"][name] = {"ids": t["ids"][:SAMPLE], key: t[key][:SAMPLE]}
    shape["counts"]["typechart_cells"] = len(b["typechart"])
    shape["counts"]["randbats_species"] = {
        k: len(v) for k, v in (("singles", b["randbats"]["singles"]),
                               ("doubles", b["randbats"]["doubles"]))}
    keys = list(b["randbats"]["singles"])[:SAMPLE]
    shape["samples"]["randbats"] = {k: b["randbats"]["singles"][k] for k in keys}
    OUT_SHAPE.write_text(json.dumps(shape, indent=1))
    print(f"shape: {OUT_SHAPE.stat().st_size//1024} KB")

    # ---- html viewer ---------------------------------------------------------
    html = TEMPLATE.replace("__DATA__", raw)
    OUT_HTML.write_text(html)
    print(f"viewer: {OUT_HTML.stat().st_size//1024} KB")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gamebundle viewer</title>
<style>
  :root { --bg:#0d1117; --fg:#c9d1d9; --dim:#8b949e; --acc:#58a6ff; --line:#21262d;
          --chip:#161b22; --good:#3fb950; --bad:#f85149; --mid:#d29922; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
         background:var(--bg); color:var(--fg); display:flex; height:100vh; }
  #side { width:300px; border-right:1px solid var(--line); display:flex;
          flex-direction:column; }
  #side header { padding:10px 12px; border-bottom:1px solid var(--line); }
  #q { width:100%; padding:6px 8px; background:var(--chip); color:var(--fg);
       border:1px solid var(--line); border-radius:6px; }
  #tabs { display:flex; gap:4px; padding:8px 12px 0; flex-wrap:wrap; }
  #tabs button { background:var(--chip); color:var(--dim); border:1px solid var(--line);
       border-radius:6px; padding:3px 8px; cursor:pointer; font:inherit; font-size:12px; }
  #tabs button.on { color:var(--acc); border-color:var(--acc); }
  #list { flex:1; overflow-y:auto; padding:8px 6px; }
  .it { padding:3px 8px; border-radius:5px; cursor:pointer; white-space:nowrap;
        overflow:hidden; text-overflow:ellipsis; }
  .it:hover { background:var(--chip); }
  .it.sel { background:#1f6feb33; color:var(--acc); }
  .it small { color:var(--dim); margin-left:6px; }
  main { flex:1; overflow:auto; padding:20px 26px; }
  h1 { font-size:16px; margin:0 0 4px; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:16px; }
  table { border-collapse:collapse; margin:10px 0 22px; }
  td,th { border:1px solid var(--line); padding:3px 9px; text-align:left;
          font-size:13px; }
  th { color:var(--dim); font-weight:normal; background:var(--chip); }
  .k { color:var(--dim); }
  .chip { display:inline-block; background:var(--chip); border:1px solid var(--line);
          border-radius:10px; padding:1px 9px; margin:2px 4px 2px 0; font-size:12px; }
  #chart td { width:34px; height:26px; text-align:center; cursor:default; }
  #chart td.x0 { background:#f8514922; color:var(--bad); }
  #chart td.x2 { background:#d2992218; color:var(--mid); }
  #chart td.x8 { background:#3fb95018; color:var(--good); }
  #chart td.x16 { background:#3fb95030; color:var(--good); font-weight:bold; }
  #chart td.h { background:var(--chip); color:var(--dim); font-size:11px; }
  #chart td.hh { background:var(--chip); color:var(--acc); font-size:11px; }
</style>
</head>
<body>
<div id="side">
  <header><input id="q" placeholder="search ids… (e.g. garchomp, roost)" autofocus></header>
  <div id="tabs"></div>
  <div id="list"></div>
</div>
<main id="main"></main>
<script>
const B = __DATA__;
const T = B.vocab.types, CAT = B.vocab.categories, TG = B.vocab.targets;
const tabs = [["species","species"],["moves","moves"],["items","items"],
              ["abilities","abilities"],["sets","randbats"],["chart","typechart"]];
let cur = "species", sel = 0;

function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}

function decodeSpecies(i){
  const r = B.species.rows[i];
  const out = `<h1>${B.species.ids[i]}</h1><div class="sub">species #${i}</div><table>`;
  out += `<tr><th>types</th><td>${r[0]>=0?T[r[0]]:"-"}${r[1]>=0?" / "+T[r[1]]:""}</td></tr>`;
  ["hp","atk","def","spa","spd","spe"].forEach((k,j)=>out+=`<tr><th>${k}</th><td>${r[2+j]}</td></tr>`);
  out += `<tr><th>abilities</th><td>${r.slice(8,11).map(a=>a>=0?esc(B.abilities.ids[a]):"-").join(", ")}</td></tr></table>`;
  out += setsHtml(i);
  return out;
}
function setsHtml(i){
  let out = "";
  for (const fmt of ["singles","doubles"]){
    const e = B.randbats[fmt][String(i)];
    if (!e) continue;
    out += `<h1>${fmt} sets <span class="sub">(level ${e.level})</span></h1>`;
    for (const s of e.sets){
      const role = s[0]>=0 ? B.vocab.roles[s[0]] : "?";
      out += `<div><span class="chip">${esc(role)}</span>` +
        s[1].map(m=>`<span class="chip">${esc(B.moves.ids[m])}</span>`).join("") + "</div>" +
        `<div class="sub" style="margin:4px 0 14px">abilities: ${s[2].map(a=>esc(B.abilities.ids[a])).join(", ") || "-"} · tera: ${s[3].map(t=>T[t]).join(", ") || "-"} · evs ${s[4].join("/")}</div>`;
    }
  }
  return out || `<div class="sub">no randbats sets</div>`;
}
function decodeMove(i){
  const r = B.moves.rows[i];
  return `<h1>${B.moves.ids[i]}</h1><div class="sub">move #${i}</div><table>` +
    `<tr><th>type</th><td>${T[r[0]]}</td></tr><tr><th>category</th><td>${CAT[r[1]]}</td></tr>` +
    `<tr><th>power</th><td>${r[2]||"-"}</td></tr><tr><th>accuracy</th><td>${r[3]||"always"}</td></tr>` +
    `<tr><th>pp</th><td>${r[4]||"-"}</td></tr><tr><th>priority</th><td>${r[5]||0}</td></tr>` +
    `<tr><th>target</th><td>${r[6]>=0?esc(TG[r[6]]):"-"}</td></tr></table>`;
}
function decodeTags(kind, i){
  const t = B[kind], v = B.vocab.tags[kind === "items" ? "item" : "ability"];
  return `<h1>${t.ids[i]}</h1><div class="sub">${kind.slice(0,-1)} #${i}</div>` +
    t.tags[i].map(x=>`<span class="chip">${esc(v[x])}</span>`).join("") +
    (t.tags[i].length ? "" : `<div class="sub">no tags</div>`);
}
function chartHtml(){
  let out = `<h1>typechart</h1><div class="sub">multiplier x4 encoding · hover row header = attacker</div><table id="chart"><tr><td class="h">atk\\def</td>`;
  T.forEach(t=>out+=`<td class="h">${t.slice(0,4)}</td>`);
  out += "</tr>";
  T.forEach((at,i)=>{
    out += `<tr><td class="hh">${at}</td>`;
    T.forEach((dt,j)=>{
      const x = B.typechart[i*19+j];
      out += `<td class="x${x}" title="${at} -> ${dt}: ${x/4}x">${x===4?"":x/4}</td>`;
    });
    out += "</tr>";
  });
  return out + "</table>";
}
function setsTabHtml(){
  const n = B.species.ids.length;
  return `<h1>randbats sets</h1><div class="sub">${Object.keys(B.randbats.singles).length} species with sets — pick any species from the sidebar</div>` +
    chartlessSample();
}
function chartlessSample(){ return ""; }

function renderList(){
  const q = document.getElementById("q").value.toLowerCase();
  let ids = [];
  if (cur === "species") ids = B.species.ids;
  else if (cur === "moves") ids = B.moves.ids;
  else if (cur === "items") ids = B.items.ids;
  else if (cur === "abilities") ids = B.abilities.ids;
  const list = document.getElementById("list");
  list.innerHTML = "";
  if (cur === "chart"){ document.getElementById("main").innerHTML = chartHtml(); return; }
  if (cur === "sets"){ document.getElementById("main").innerHTML = setsTabHtml(); return; }
  const hit = [];
  for (let i=0;i<ids.length && hit.length<400;i++)
    if (!q || ids[i].includes(q)) hit.push(i);
  hit.forEach(i=>{
    const d = document.createElement("div");
    d.className = "it" + (i===sel?" sel":"");
    d.textContent = ids[i];
    d.onclick = ()=>{ sel = i; renderMain(); renderList(); };
    list.appendChild(d);
  });
  if (hit.length === 400){
    const d = document.createElement("div");
    d.className = "sub"; d.style.padding = "6px 10px";
    d.textContent = "…400 shown, keep typing";
    list.appendChild(d);
  }
  renderMain();
}
function renderMain(){
  const m = document.getElementById("main");
  if (cur === "species") m.innerHTML = decodeSpecies(sel);
  else if (cur === "moves") m.innerHTML = decodeMove(sel);
  else if (cur === "items") m.innerHTML = decodeTags("items", sel);
  else if (cur === "abilities") m.innerHTML = decodeTags("abilities", sel);
}
const tabsEl = document.getElementById("tabs");
tabs.forEach(([label, key])=>{
  const b = document.createElement("button");
  b.textContent = label;
  if (key === cur) b.classList.add("on");
  b.onclick = ()=>{
    cur = key; sel = 0;
    [...tabsEl.children].forEach(c=>c.classList.remove("on"));
    b.classList.add("on");
    renderList();
  };
  tabsEl.appendChild(b);
});
document.getElementById("q").addEventListener("input", renderList);
renderList();
</script>
</body>
</html>"""

if __name__ == "__main__":
    sys.exit(main())
