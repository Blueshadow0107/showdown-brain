"""build_viewer.py — emit a self-contained HTML viewer + a JSON Crack shape file
for assets/gamebundle.json.

  assets/gamebundle.html   tabs: search/decoded-rows, interactive typechart,
                           and a CANVAS KNOWLEDGE GRAPH (species -> types/
                           abilities/set-moves, moves -> type, items/abilities
                           -> tags) with force-layout physics, hover
                           neighborhoods, and click-to-decode. no deps.
  assets/gamebundle.shape.json   structure-only summary for JSON Crack's cap
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "assets" / "gamebundle.json"
OUT_HTML = ROOT / "assets" / "gamebundle.html"
OUT_SHAPE = ROOT / "assets" / "gamebundle.shape.json"

SAMPLE = 3


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
  #side { width:320px; border-right:1px solid var(--line); display:flex;
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
  main { flex:1; overflow:auto; padding:20px 26px; position:relative; }
  h1 { font-size:16px; margin:0 0 4px; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:16px; }
  table { border-collapse:collapse; margin:10px 0 22px; }
  td,th { border:1px solid var(--line); padding:3px 9px; text-align:left; font-size:13px; }
  th { color:var(--dim); font-weight:normal; background:var(--chip); }
  .chip { display:inline-block; background:var(--chip); border:1px solid var(--line);
          border-radius:10px; padding:1px 9px; margin:2px 4px 2px 0; font-size:12px; }
  #chart td { width:34px; height:26px; text-align:center; cursor:default; }
  #chart td.x0 { background:#f8514922; color:var(--bad); }
  #chart td.x2 { background:#d2992218; color:var(--mid); }
  #chart td.x8 { background:#3fb95018; color:var(--good); }
  #chart td.x16 { background:#3fb95030; color:var(--good); font-weight:bold; }
  #chart td.h { background:var(--chip); color:var(--dim); font-size:11px; }
  #chart td.hh { background:var(--chip); color:var(--acc); font-size:11px; }
  #gwrap { position:absolute; inset:0; display:none; }
  #graph { width:100%; height:100%; display:block; cursor:grab; }
  #ginfo { position:absolute; left:12px; bottom:10px; color:var(--dim);
           font-size:11px; pointer-events:none; }
  #ghint { position:absolute; right:12px; top:10px; color:var(--dim);
           font-size:11px; pointer-events:none; text-align:right; }
  #gpanel { position:absolute; right:0; top:0; bottom:0; width:330px; overflow-y:auto;
            background:var(--bg); border-left:1px solid var(--line); padding:14px;
            display:none; }
  #gpanel h1 { font-size:15px; }
</style>
</head>
<body>
<div id="side">
  <header><input id="q" placeholder="search ids… (e.g. garchomp, roost)" autofocus></header>
  <div id="tabs"></div>
  <div id="list"></div>
</div>
<main id="main">
  <div id="gwrap"><canvas id="graph"></canvas>
    <div id="ginfo"></div>
    <div id="ghint">drag = pan · wheel = zoom · drag node = pin<br>hover = neighbourhood · click = decode</div>
    <div id="gpanel"></div>
  </div>
</main>
<script>
const B = __DATA__;
const T = B.vocab.types, CAT = B.vocab.categories, TG = B.vocab.targets;
const tabs = [["species","species"],["moves","moves"],["items","items"],
              ["abilities","abilities"],["sets","randbats"],["chart","typechart"],
              ["graph","graph"]];
let cur = "species", sel = 0;

function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
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
function decodeSpecies(i){
  const r = B.species.rows[i];
  let out = `<h1>${B.species.ids[i]}</h1><div class="sub">species #${i}</div><table>`;
  out += `<tr><th>types</th><td>${r[0]>=0?T[r[0]]:"-"}${r[1]>=0?" / "+T[r[1]]:""}</td></tr>`;
  ["hp","atk","def","spa","spd","spe"].forEach((k,j)=>out+=`<tr><th>${k}</th><td>${r[2+j]}</td></tr>`);
  out += `<tr><th>abilities</th><td>${r.slice(8,11).map(a=>a>=0?esc(B.abilities.ids[a]):"-").join(", ")}</td></tr></table>`;
  return out + setsHtml(i);
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
  let out = `<h1>typechart</h1><div class="sub">multiplier x4 encoding</div><table id="chart"><tr><td class="h">atk\\def</td>`;
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
function renderMain(){
  const m = document.getElementById("main");
  if (cur === "species") m.innerHTML = decodeSpecies(sel);
  else if (cur === "moves") m.innerHTML = decodeMove(sel);
  else if (cur === "items") m.innerHTML = decodeTags("items", sel);
  else if (cur === "abilities") m.innerHTML = decodeTags("abilities", sel);
}
function renderList(){
  document.getElementById("gwrap").style.display = cur === "graph" ? "block" : "none";
  document.getElementById("main").childNodes.forEach(n=>{
    if (n.nodeType===1 && n.id!=="gwrap") n.remove();
  });
  if (cur === "graph"){ Graph.enter(); return; }
  Graph.exit();
  const q = document.getElementById("q").value.toLowerCase();
  let ids = [];
  if (cur === "species") ids = B.species.ids;
  else if (cur === "moves") ids = B.moves.ids;
  else if (cur === "items") ids = B.items.ids;
  else if (cur === "abilities") ids = B.abilities.ids;
  const list = document.getElementById("list");
  list.innerHTML = "";
  if (cur === "chart"){
    const d = document.createElement("div"); d.innerHTML = chartHtml();
    m.appendChild(d); return;
  }
  if (cur === "sets"){
    const d = document.createElement("div");
    d.innerHTML = `<h1>randbats sets</h1><div class="sub">pick any species in the species tab (sets are on its page)</div>`;
    document.getElementById("main").appendChild(d); return;
  }
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

/* ================= knowledge graph ================= */
const Graph = (() => {
  const COLORS = {species:"#58a6ff", moves:"#3fb950", abilities:"#d29922",
                  items:"#bc8cff", types:"#f778ba", tags:"#8b949e"};
  let nodes = [], edges = [], adj = [], canvas, ctx, running = false;
  let cam = {x:0, y:0, z:0.35}, hover = -1, focus = -1, dragged = -1;
  let W = 0, H = 0;

  function build(){
    nodes = []; edges = [];
    const idx = {};
    const add = (table, i) => {
      const key = table + ":" + i;
      if (idx[key] === undefined){
        idx[key] = nodes.length;
        nodes.push({table, i, x:(Math.random()-0.5)*1600, y:(Math.random()-0.5)*1600,
                    vx:0, vy:0, pin:false});
      }
      return idx[key];
    };
    const link = (a,b) => { if (a!==b) edges.push([a,b]); };
    // types
    T.forEach((_,i)=>add("types", i));
    B.vocab.tags.item.forEach((_,i)=>add("tags", i));
    B.vocab.tags.ability.forEach((_,i)=>add("tags", 100+i));
    // species -> types, abilities
    B.species.rows.forEach((r,s)=>{
      const ns = add("species", s);
      if (r[0]>=0) link(ns, add("types", r[0]));
      if (r[1]>=0) link(ns, add("types", r[1]));
      r.slice(8,11).forEach(a=>{ if (a>=0) link(ns, add("abilities", a)); });
    });
    // species -> moves (via all its sets), moves -> type
    const moveType = B.moves.rows.map(r=>r[0]);
    for (const fmt of ["singles","doubles"]){
      for (const k in B.randbats[fmt]){
        const s = +k;
        const ns = idx["species:"+s];
        if (ns === undefined) continue;
        for (const st of B.randbats[fmt][k].sets)
          for (const m of st[1]) link(ns, add("moves", m));
      }
    }
    B.moves.rows.forEach((r,m)=>{ if (r[0]>=0) link(idx["moves:"+m], idx["types:"+r[0]]); });
    // items/abilities -> tags
    B.items.tags.forEach((ts,i)=>ts.forEach(t=>link(add("items",i), idx["tags:"+t])));
    B.abilities.tags.forEach((ts,i)=>ts.forEach(t=>link(add("abilities",i), idx["tags:"+(100+t)])));
    // adjacency
    adj = nodes.map(()=>[]);
    edges.forEach(([a,b])=>{ adj[a].push(b); adj[b].push(a); });
  }

  function label(n){
    if (n.table === "species") return B.species.ids[n.i];
    if (n.table === "moves") return B.moves.ids[n.i];
    if (n.table === "abilities") return B.abilities.ids[n.i];
    if (n.table === "items") return B.items.ids[n.i];
    if (n.table === "types") return T[n.i];
    const t = n.i >= 100 ? B.vocab.tags.ability[n.i-100] : B.vocab.tags.item[n.i];
    return "#" + t;
  }

  function step(){
    // spatial hash for repulsion
    const cell = 60, grid = new Map();
    nodes.forEach((n,i)=>{
      const k = ((n.x/cell)|0) + ":" + ((n.y/cell)|0);
      (grid.get(k) || grid.set(k, []).get(k)).push(i);
    });
    const R2 = 60*60;
    nodes.forEach((n,i)=>{
      if (n.pin) return;
      const cx = (n.x/cell)|0, cy = (n.y/cell)|0;
      let fx = 0, fy = 0;
      for (let gx=cx-1; gx<=cx+1; gx++) for (let gy=cy-1; gy<=cy+1; gy++){
        const bucket = grid.get(gx+":"+gy); if (!bucket) continue;
        for (const j of bucket){
          if (j===i) continue;
          const o = nodes[j], dx = n.x-o.x, dy = n.y-o.y, d2 = dx*dx+dy*dy;
          if (d2 < R2 && d2 > 0.01){ const f = 26/d2; fx += dx*f; fy += dy*f; }
        }
      }
      for (const j of adj[i]){
        const o = nodes[j], dx = o.x-n.x, dy = o.y-n.y, d = Math.sqrt(dx*dx+dy*dy)||1;
        const f = (d-70)*0.004; fx += dx/d*f*70; fy += dy/d*f*70;
      }
      n.vx = (n.vx+fx)*0.6; n.vy = (n.vy+fy)*0.6;
      n.x += Math.max(-6,Math.min(6,n.vx)); n.y += Math.max(-6,Math.min(6,n.vy));
    });
  }

  function draw(){
    ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,W,H);
    ctx.translate(W/2,H/2); ctx.scale(cam.z,cam.z); ctx.translate(cam.x,cam.y);
    const z = cam.z;
    ctx.lineWidth = 1/z;
    ctx.strokeStyle = "#21262d";
    ctx.beginPath();
    for (const [a,b] of edges){
      ctx.moveTo(nodes[a].x, nodes[a].y); ctx.lineTo(nodes[b].x, nodes[b].y);
    }
    ctx.stroke();
    const near = new Set();
    if (hover>=0){ near.add(hover); adj[hover].forEach(j=>near.add(j)); }
    if (focus>=0){ near.add(focus); adj[focus].forEach(j=>near.add(j)); }
    nodes.forEach((n,i)=>{
      const r = (n.table==="types"||n.table==="tags") ? 3.5 : 2.6;
      ctx.fillStyle = COLORS[n.table];
      ctx.globalAlpha = (near.size && !near.has(i)) ? 0.08 : 0.9;
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 7); ctx.fill();
    });
    ctx.globalAlpha = 1;
    if (z > 1.1 || near.size){
      ctx.font = `${11/z}px ui-monospace,monospace`;
      ctx.fillStyle = "#c9d1d9";
      const seen = new Set();
      nodes.forEach((n,i)=>{
        if (!(near.has(i) || z > 1.6)) return;
        const key = ((n.x/90)|0)+":"+((n.y/14)|0);
        if (seen.has(key)) return; seen.add(key);
        ctx.fillText(label(n), n.x+4/z*2, n.y+3);
      });
    }
  }

  function loop(){
    if (!running) return;
    for (let k=0;k<3;k++) step();
    draw();
    requestAnimationFrame(loop);
  }

  function toWorld(mx,my){ return [(mx-W/2)/cam.z - cam.x, (my-H/2)/cam.z - cam.y]; }

  function bind(){
    canvas.addEventListener("mousedown", e=>{
      const [wx,wy] = toWorld(e.offsetX,e.offsetY);
      let best=-1, bd=20/cam.z;
      nodes.forEach((n,i)=>{ const d=Math.hypot(n.x-wx,n.y-wy); if(d<bd){bd=d;best=i;} });
      if (best>=0){ dragged = best; nodes[best].pin = true; }
      canvas.style.cursor = best>=0 ? "move" : "grabbing";
    });
    window.addEventListener("mouseup", ()=>{
      if (dragged>=0) nodes[dragged].pin = false;
      dragged = -1; canvas.style.cursor = "grab";
    });
    canvas.addEventListener("mousemove", e=>{
      const [wx,wy] = toWorld(e.offsetX,e.offsetY);
      if (dragged>=0){ nodes[dragged].x = wx; nodes[dragged].y = wy; return; }
      let best=-1, bd=16/cam.z;
      nodes.forEach((n,i)=>{ const d=Math.hypot(n.x-wx,n.y-wy); if(d<bd){bd=d;best=i;} });
      hover = best;
      canvas.title = best>=0 ? label(nodes[best]) : "";
    });
    canvas.addEventListener("click", e=>{
      if (hover<0){ document.getElementById("gpanel").style.display="none"; focus=-1; return; }
      focus = hover;
      const n = nodes[hover];
      const p = document.getElementById("gpanel");
      p.style.display = "block";
      p.innerHTML = n.table==="species" ? decodeSpecies(n.i)
        : n.table==="moves" ? decodeMove(n.i)
        : (n.table==="items"||n.table==="abilities") ? decodeTags(n.table, n.i)
        : `<h1>${esc(label(n))}</h1>`;
    });
    canvas.addEventListener("wheel", e=>{
      e.preventDefault();
      cam.z = Math.max(0.08, Math.min(6, cam.z * (e.deltaY<0 ? 1.15 : 0.87)));
    }, {passive:false});
    let pan = null;
    canvas.addEventListener("mousedown", e=>{ if(dragged<0) pan = {x:e.clientX,y:e.clientY}; });
    window.addEventListener("mousemove", e=>{
      if (pan && dragged<0){
        cam.x += (e.clientX-pan.x)/cam.z; cam.y += (e.clientY-pan.y)/cam.z;
        pan = {x:e.clientX,y:e.clientY};
      }
    });
    window.addEventListener("mouseup", ()=>pan=null);
    document.getElementById("q").addEventListener("input", ()=>{
      if (cur!=="graph") return;
      const q = document.getElementById("q").value.toLowerCase();
      if (!q) return;
      const found = nodes.findIndex(n=>label(n).toLowerCase().includes(q));
      if (found>=0){
        focus = found;
        cam.x = -nodes[found].x; cam.y = -nodes[found].y; cam.z = Math.max(cam.z, 1.2);
        const n = nodes[found];
        const p = document.getElementById("gpanel");
        p.style.display = "block";
        p.innerHTML = n.table==="species" ? decodeSpecies(n.i)
          : n.table==="moves" ? decodeMove(n.i)
          : (n.table==="items"||n.table==="abilities") ? decodeTags(n.table, n.i)
          : `<h1>${esc(label(n))}</h1>`;
      }
    });
  }

  function enter(){
    if (!nodes.length){ build(); }
    canvas = document.getElementById("graph");
    canvas.width = W = canvas.clientWidth * devicePixelRatio;
    canvas.height = H = canvas.clientHeight * devicePixelRatio;
    ctx = canvas.getContext("2d");
    document.getElementById("ginfo").textContent =
      `${nodes.length} nodes · ${edges.length} edges`;
    if (!running){ running = true; bind(); loop(); }
  }
  function exit(){ running = false; }
  return {enter, exit};
})();

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
