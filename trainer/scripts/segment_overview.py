"""Builds a local page that shows, for each stem and instrument, how much of the stem is that instrument alone.

    python scripts/segment_overview.py [--scores C:/workarea/segment-candidates/scores.json] 

`scores.json` holds, per song, the AudioSet tagger's scores (5 s windows, hop 2.5 s) of instrument families over the whole
`other` stem. The page `overview.html` (next to it) colours every window of the stem:

  green  the instrument is heard and nothing else is (score >= `target` and every other family < `other`)
  amber  the instrument is heard together with another family
  red    another family is heard and the instrument is not
  grey   nothing is recognised (silence, or something the tagger has no name for)

so a stem that is entirely piano is one green bar, and a stem with short vibraphone parts shows them as breaks, listed
as interruptions with the family that caused them. Click a bar to play from there (audio: the `audio` folder of WAV copies, see segment_review.py). Judge each stem `all`, `mostly`, `partly`
or `no`; judgements are kept in the browser and "Save CSV" downloads `stems.csv` with the judgement, the share of clean
windows and the interruptions. The thresholds can be changed on the page.
"""

import argparse
import json
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stem overview</title>
<style>
:root { --bg:#fff; --fg:#1b1f24; --muted:#5b6570; --card:#f4f6f8; --line:#d5dbe1; --accent:#1f6feb; --clean:#2da44e; --mixed:#d4a72c; --other:#e5534b; --quiet:#c9d1d9; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1318; --fg:#e6eaee; --muted:#9aa5b1; --card:#171d24; --line:#2a323c; --accent:#58a6ff; --clean:#3fb950; --mixed:#d29922; --other:#f85149; --quiet:#3b444f; } }
body { margin:0; font:15px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; z-index:2; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
main { padding:8px 16px 60px; max-width:960px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:12px 0; }
.card h2 { font-size:16px; margin:0 0 2px; } .meta { color:var(--muted); font-size:13px; }
.bar { display:flex; height:30px; margin:8px 0 4px; border-radius:4px; overflow:hidden; cursor:pointer; border:1px solid var(--line); }
.bar span { flex:1; min-width:1px; } .c-clean{background:var(--clean)} .c-mixed{background:var(--mixed)} .c-other{background:var(--other)} .c-quiet{background:var(--quiet)}
.legend span { display:inline-block; width:10px; height:10px; margin:0 3px 0 10px; border-radius:2px; }
button, input { font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:4px 10px; cursor:pointer; }
input { cursor:text; width:64px; } button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:6px; }
.gaps { font-size:13px; color:var(--muted); margin-top:4px; } .gaps a { color:var(--accent); cursor:pointer; margin-right:10px; white-space:nowrap; }
</style></head><body>
<header>
  <strong>Stem overview</strong>
  <label>instrument alone: score &ge; <input type="number" id="tt" step="0.05" value="0.3"></label>
  <label>other family: score &ge; <input type="number" id="to" step="0.05" value="0.2"></label>
  <button id="stop">Stop</button><button id="save" class="on">Save CSV</button><span id="count" class="meta"></span>
  <span class="legend meta"><span class="c-clean"></span>only this<span class="c-mixed"></span>with others<span class="c-other"></span>others only<span class="c-quiet"></span>nothing named</span>
</header>
<main id="list"></main>
<audio id="audio" preload="auto"></audio>
<script>
const DATA = __DATA__, BASE = __BASE__, HOP = 2.5, WIN = 5.0, KEY = "segment-overview-v1";
let saved = {}; try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} };
const audio = document.getElementById("audio");
const url = d => BASE + encodeURIComponent(d.split + " - " + d.song + ".wav");
function playFrom(d, t) {
  const go = () => { audio.currentTime = Math.max(0, t); audio.play(); }, u = url(d);
  if (audio.dataset.u !== u) { audio.dataset.u = u; audio.src = u; audio.addEventListener("loadedmetadata", go, { once: true }); audio.load(); } else go();
}
document.getElementById("stop").onclick = () => audio.pause();
// the window i covers [i*HOP, i*HOP+WIN]; its middle HOP seconds are what it is used for
const span = i => [i * HOP + (WIN - HOP) / 2, i * HOP + (WIN + HOP) / 2];
function analyse(d, inst, tt, to) {
  const fams = Object.keys(d.scores), n = d.scores[inst].length, cells = [];
  for (let i = 0; i < n; i++) {
    const s = d.scores[inst][i]; let best = "", bs = 0;
    for (const f of fams) if (f !== inst && d.scores[f][i] > bs) { bs = d.scores[f][i]; best = f; }
    const hasT = s >= tt, hasO = bs >= to;
    cells.push({ k: hasT ? (hasO ? "mixed" : "clean") : (hasO ? "other" : "quiet"), fam: hasO ? best : "" });
  }
  const gaps = []; let cur = null;
  cells.forEach((c, i) => {
    if (c.k !== "clean") { const [a, b] = span(i); if (cur && cur.fam === (c.fam || "?") && a - cur.end < 0.1) cur.end = b; else { cur = { start: a, end: b, fam: c.fam || "?" }; gaps.push(cur); } } else cur = null;
  });
  const share = k => cells.filter(c => c.k === k).length / n;
  return { cells, gaps, clean: share("clean") };
}
function render() {
  const tt = +document.getElementById("tt").value, to = +document.getElementById("to").value;
  const list = document.getElementById("list"); list.innerHTML = ""; let done = 0, total = 0;
  const items = [];
  for (const d of DATA) for (const inst of ["piano", "organ"]) if (d.tags.split("+").includes(inst)) items.push({ d, inst, a: analyse(d, inst, tt, to) });
  items.sort((x, y) => y.a.clean - x.a.clean);
  for (const { d, inst, a } of items) {
    const id = d.split + "|" + d.song + "|" + inst, v = saved[id] || ""; total++; if (v) done++;
    const card = document.createElement("div"); card.className = "card";
    const pct = k => Math.round(100 * a.cells.filter(c => c.k === k).length / a.cells.length);
    const gaps = a.gaps.filter(g => g.end - g.start >= 0).slice(0, 12);
    card.innerHTML = `<h2>${d.song} &middot; <b>${inst}</b> <span class="meta">(${d.split}, ${Math.round(a.cells.length * HOP)} s, your tags: ${d.tags.replaceAll("+", ", ")})</span></h2>
      <div class="meta">only ${inst}: <b>${pct("clean")}%</b> &middot; with others: ${pct("mixed")}% &middot; others only: ${pct("other")}% &middot; nothing named: ${pct("quiet")}%</div>
      <div class="bar">${a.cells.map((c, i) => `<span class="c-${c.k}" title="${span(i)[0].toFixed(0)} s ${c.fam}"></span>`).join("")}</div>
      <div class="gaps">${a.gaps.length ? a.gaps.length + " interruption(s): " : "no interruptions"}${gaps.map(g => `<a data-t="${g.start}">${g.start.toFixed(0)}-${g.end.toFixed(0)} s ${g.fam}</a>`).join("")}${a.gaps.length > 12 ? "&hellip;" : ""}</div>
      <div class="row">${["all", "mostly", "partly", "no"].map(x => `<button data-v="${x}" class="${v === x ? "on" : ""}">${x}</button>`).join("")}
      <span class="meta">all = the whole stem is ${inst} &middot; mostly = ${inst} with short other parts &middot; partly = only some stretches &middot; no</span></div>`;
    card.querySelector(".bar").onclick = e => { const r = e.currentTarget.getBoundingClientRect(); playFrom(d, (e.clientX - r.left) / r.width * a.cells.length * HOP); };
    card.querySelectorAll(".gaps a").forEach(el => el.onclick = () => playFrom(d, +el.dataset.t - 2));
    card.querySelectorAll(".row button").forEach(b => b.onclick = () => { saved[id] = b.dataset.v; persist(); render(); });
    list.appendChild(card);
  }
  document.getElementById("count").textContent = done + " / " + total + " judged";
  window._items = items;
}
document.getElementById("tt").onchange = render; document.getElementById("to").onchange = render;
document.getElementById("save").onclick = () => {
  const q = x => '"' + String(x).replaceAll('"', '""') + '"';
  const rows = ["split,song,stem,instrument,verdict,clean_pct,interruptions"];
  for (const { d, inst, a } of window._items) { const v = saved[d.split + "|" + d.song + "|" + inst]; if (!v) continue;
    rows.push([q(d.split), q(d.song), "other", inst, v, Math.round(100 * a.clean), q(a.gaps.map(g => g.start.toFixed(1) + "-" + g.end.toFixed(1) + ":" + g.fam).join(";"))].join(",")); }
  const el = document.createElement("a"); el.href = URL.createObjectURL(new Blob(["\ufeff" + rows.join("\n") + "\n"], { type: "text/csv" }));
  el.download = "stems.csv"; el.click();
};
render();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(r"C:\workarea\segment-candidates\scores.json"))
    args = parser.parse_args()
    data = json.loads(args.scores.read_text(encoding="utf-8"))
    out = args.scores.parent / "overview.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__BASE__", json.dumps("audio/")), encoding="utf-8")
    print(f"{len(data)} songs -> {out}")


if __name__ == "__main__":
    main()
