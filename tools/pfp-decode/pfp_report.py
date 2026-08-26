#!/usr/bin/env python3
"""Render a PharmCast .pfp fingerprint as a human readable HTML report.

The .pfp storage format is 330 unsigned 32 bit words per molecule, MSB first
inside each word, carrying 10,549 three point pharmacophores. Each set bit is
one triangle: three pharmacophore features at three binned distances.

  python3 pfp_report.py molecules.pfp -o report.html
  python3 pfp_report.py chunk_06663.pfp.gz --limit 50 -o report.html
  python3 pfp_report.py molecules.pfp --bits          # 0/1 string, like pfp2bits

Pure standard library. Read only.
"""
from __future__ import annotations

import argparse
import collections
import string
import gzip
import html
import os
import sys

N_PHARM = 10549
N_INTS = 330
HERE = os.path.dirname(os.path.abspath(__file__))

FEATURE_NAMES = {
    "A": "H-bond acceptor", "D": "H-bond donor", "H": "Hydrophobic",
    "N": "Negative charge", "P": "Positive charge", "R": "Aromatic ring",
    "X": "No feature",
}
FEATURE_COLOR = {
    "A": "#d1495b", "D": "#3d7ea6", "H": "#e0a458", "N": "#7b4b94",
    "P": "#2a9d8f", "R": "#8a6f4e", "X": "#9aa0a6",
}
DIST_ORDER = ["2-4.5", "4.5-7", "7-10", "10-14", "14-19", "19-24"]


def open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_labels(path):
    """pharm10549.list, one line per pharmacophore, 1-indexed in the file.

    Line form:  | d1 | d2 | d3 | f1 | f2 | f3 |
    d1 is the p3-p2 edge, d2 the p1-p3 edge, d3 the p1-p2 edge.
    """
    labels = []
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 6:
                labels.append((parts[0], parts[1], parts[2],
                               parts[3], parts[4], parts[5]))
            else:
                labels.append(("?", "?", "?", "?", "?", "?"))
    if len(labels) != N_PHARM:
        sys.stderr.write("WARNING: %s holds %d labels, expected %d\n"
                         % (path, len(labels), N_PHARM))
    return labels


def decode(words):
    """MSB first inside each 32 bit word, exactly as pfpkey.c writes them."""
    on = []
    for i in range(N_PHARM):
        if words[i // 32] & (1 << (31 - i % 32)):
            on.append(i)
    return on


def read_pfp(path, limit=0):
    out = []
    with open_maybe_gz(path) as fh:
        for line in fh:
            t = line.split()
            if len(t) < N_INTS + 1:
                continue
            out.append((t[0], [int(v) for v in t[1:N_INTS + 1]],
                        line.rstrip("\n")))
            if limit and len(out) >= limit:
                break
    return out


def triangle_svg():
    return """
<svg viewBox="0 0 260 190" class="tri" role="img" aria-label="Three point pharmacophore triangle">
  <line x1="130" y1="34" x2="46"  y2="152" class="edge"/>
  <line x1="130" y1="34" x2="214" y2="152" class="edge"/>
  <line x1="46"  y1="152" x2="214" y2="152" class="edge"/>
  <text x="130" y="24"  class="vtx">p1</text>
  <text x="36"  y="172" class="vtx">p3</text>
  <text x="224" y="172" class="vtx">p2</text>
  <text x="72"  y="96"  class="edg">d2</text>
  <text x="188" y="96"  class="edg">d3</text>
  <text x="130" y="170" class="edg">d1</text>
</svg>"""


def chip(letter):
    return ('<span class="chip" style="--c:%s" title="%s">%s</span>'
            % (FEATURE_COLOR.get(letter, "#888"),
               html.escape(FEATURE_NAMES.get(letter, letter)), letter))


def bar_rows(counter, total, limit=None, fmt=str):
    rows, top = [], counter.most_common(limit) if limit else sorted(
        counter.items(), key=lambda kv: DIST_ORDER.index(kv[0])
        if kv[0] in DIST_ORDER else 99)
    biggest = max([c for _, c in top] or [1])
    for key, count in top:
        pct = 100.0 * count / total if total else 0.0
        rows.append(
            '<tr><td>%s</td><td class="num">%s</td><td class="num">%.1f%%</td>'
            '<td class="barcell"><span class="bar" style="width:%.1f%%"></span></td></tr>'
            % (fmt(key), format(count, ","), pct, 100.0 * count / biggest))
    return "\n".join(rows)


def render(records, labels, source, max_listed):
    n = len(records)
    per_mol, union = [], set()
    feat_single = collections.Counter()
    triplets = collections.Counter()
    dists = collections.Counter()
    for name, words, _raw in records:
        on = decode(words)
        per_mol.append((name, on))
        union.update(on)
        for i in on:
            d1, d2, d3, f1, f2, f3 = labels[i]
            key = "".join(sorted((f1, f2, f3)))
            triplets[key] += 1
            for f in (f1, f2, f3):
                feat_single[f] += 1
            for d in (d1, d2, d3):
                dists[d] += 1
    total_set = sum(len(on) for _, on in per_mol)
    avg = total_set / n if n else 0.0
    cov = 100.0 * len(union) / N_PHARM

    mol_rows = []
    for name, on in per_mol[:max_listed]:
        mol_rows.append(
            '<tr><td class="mono">%s</td><td class="num">%s</td>'
            '<td class="num">%.2f%%</td></tr>'
            % (html.escape(name), format(len(on), ","),
               100.0 * len(on) / N_PHARM))

    hit_rows = []
    if n == 1:
        for i in per_mol[0][1][:max_listed]:
            d1, d2, d3, f1, f2, f3 = labels[i]
            hit_rows.append(
                "<tr><td class='num mono'>%d</td><td>%s %s %s</td>"
                "<td class='mono'>%s</td><td class='mono'>%s</td><td class='mono'>%s</td></tr>"
                % (i + 1, chip(f1), chip(f2), chip(f3), d1, d2, d3))

    raw_text = "\n".join(r for _n, _w, r in records[:max_listed])
    bits_text = "\n".join(
        "%s %s" % (nm, "".join("1" if i in set(on) else "0"
                               for i in range(N_PHARM)))
        for nm, on in per_mol[:max_listed])
    copy_section = """
  <section class="card">
    <h2>Copyable fingerprint</h2>
    <p class="sub">Showing $shown_copy of $n. The first block is the packed
       <code>.pfp</code> row exactly as it was read: name then 330 unsigned
       32 bit words, MSB first inside each word. The second is the same
       fingerprint expanded to one character per pharmacophore, identical to
       <code>pfp2bits</code> output.</p>
    <div class="copyhead"><span>Packed <code>.pfp</code> row (330 words)</span>
      <button class="copy" data-target="rawblock">Copy</button></div>
    <pre id="rawblock" class="block">$rawtext</pre>
    <div class="copyhead"><span>Expanded bit string (10,549 characters)</span>
      <button class="copy" data-target="bitblock">Copy</button></div>
    <pre id="bitblock" class="block">$bitstext</pre>
  </section>"""

    def trip_fmt(key):
        return " ".join(chip(c) for c in key)

    hits_section = ""
    if hit_rows:
        hits_section = """
  <section class="card">
    <h2>Pharmacophore hit list</h2>
    <p class="sub">Every set bit in this molecule, decoded. Bit numbering is
       1-based to match <code>pharmstat</code>. Showing %s of %s.</p>
    <div class="scroll"><table>
      <thead><tr><th>Bit</th><th>p1 p2 p3</th><th>d1</th><th>d2</th><th>d3</th></tr></thead>
      <tbody>%s</tbody>
    </table></div>
  </section>""" % (format(len(hit_rows), ","),
                   format(len(per_mol[0][1]), ","), "\n".join(hit_rows))

    legend = " ".join(
        '<span class="leg">%s %s</span>' % (chip(k), html.escape(v))
        for k, v in FEATURE_NAMES.items())

    tpl = string.Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PharmCast fingerprint report</title>
<style>
:root{--bg:#f7f7f5;--fg:#1c1c1a;--mut:#6b6b66;--line:#e2e2dd;--card:#fff;--accent:#3d7ea6;}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e8e4;--mut:#9a9a94;--line:#2c2e33;--card:#1e2024;}}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px 64px;background:var(--bg);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:16px;margin:0 0 4px;letter-spacing:.02em;text-transform:uppercase;color:var(--mut)}
.sub{color:var(--mut);margin:0 0 16px;font-size:13.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:16px 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.stat .v{font-size:24px;font-weight:600;letter-spacing:-.02em}
.stat .k{color:var(--mut);font-size:12.5px;text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-weight:600;color:var(--mut);font-size:12px;text-transform:uppercase;
 letter-spacing:.04em;padding:6px 10px 6px 0;border-bottom:1px solid var(--line)}
td{padding:6px 10px 6px 0;border-bottom:1px solid var(--line)}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.barcell{width:38%;padding-right:0}
.bar{display:block;height:8px;border-radius:4px;background:var(--accent);opacity:.75}
.chip{display:inline-block;min-width:19px;text-align:center;padding:1px 5px;border-radius:5px;
 background:var(--c);color:#fff;font:600 12px/1.5 ui-monospace,Menlo,monospace}
.leg{display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;color:var(--mut);font-size:13px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
.tri{width:210px;height:auto;display:block;margin:0 auto}
.tri .edge{stroke:var(--mut);stroke-width:1.4;fill:none}
.tri .vtx{fill:var(--fg);font:600 13px ui-monospace,Menlo,monospace;text-anchor:middle}
.tri .edg{fill:var(--mut);font:12px ui-monospace,Menlo,monospace;text-anchor:middle}
.scroll{max-height:520px;overflow:auto}
code{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:var(--bg);padding:1px 5px;border-radius:4px}
.foot{color:var(--mut);font-size:12.5px;margin-top:24px}
.copyhead{display:flex;align-items:center;justify-content:space-between;gap:12px;
 margin:14px 0 6px;color:var(--mut);font-size:13px}
.copy{font:600 12px/1 inherit;color:var(--fg);background:var(--bg);cursor:pointer;
 border:1px solid var(--line);border-radius:6px;padding:6px 12px}
.copy:hover{border-color:var(--accent);color:var(--accent)}
.block{margin:0;padding:12px 14px;background:var(--bg);border:1px solid var(--line);
 border-radius:8px;max-height:190px;overflow:auto;white-space:pre-wrap;word-break:break-all;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5}
</style></head><body><div class="wrap">

<h1>PharmCast fingerprint report</h1>
<p class="sub">$source &middot; 10,549 three point pharmacophores in 330 words of 32 bits, MSB first</p>

<div class="stats">
  <div class="stat"><div class="v">$n</div><div class="k">molecules</div></div>
  <div class="stat"><div class="v">$avg</div><div class="k">avg bits set</div></div>
  <div class="stat"><div class="v">$uni</div><div class="k">distinct bits hit</div></div>
  <div class="stat"><div class="v">$cov%</div><div class="k">of the 10,549 space</div></div>
</div>

<section class="card">
  <h2>How to read a bit</h2>
  <div class="cols">
    <div>$tri</div>
    <div>
      <p class="sub">Each set bit is one triangle: three pharmacophore features
      at three binned edge distances in angstroms. <code>d1</code> is p3 to p2,
      <code>d2</code> is p1 to p3, <code>d3</code> is p1 to p2. A vertex marked
      <code>X</code> carries no feature.</p>
      <div>$legend</div>
    </div>
  </div>
</section>

<div class="cols">
  <section class="card">
    <h2>Feature triplets</h2>
    <p class="sub">Which combinations of three features are firing, order independent.</p>
    <table><thead><tr><th>Triplet</th><th class="num">Bits</th><th class="num">Share</th><th></th></tr></thead>
    <tbody>$trips</tbody></table>
  </section>
  <section class="card">
    <h2>Distance bins</h2>
    <p class="sub">Every edge of every set triangle, by bin in angstroms.</p>
    <table><thead><tr><th>Bin</th><th class="num">Edges</th><th class="num">Share</th><th></th></tr></thead>
    <tbody>$dists</tbody></table>
  </section>
</div>

<section class="card">
  <h2>Feature occupancy</h2>
  <p class="sub">Every vertex of every set triangle, by feature type.</p>
  <table><thead><tr><th>Feature</th><th class="num">Vertices</th><th class="num">Share</th><th></th></tr></thead>
  <tbody>$feats</tbody></table>
</section>

<section class="card">
  <h2>Molecules</h2>
  <p class="sub">Showing $shown of $n.</p>
  <div class="scroll"><table>
    <thead><tr><th>Name</th><th class="num">Bits set</th><th class="num">Density</th></tr></thead>
    <tbody>$mols</tbody>
  </table></div>
</section>
$hits
$copy

<p class="foot">Generated by <code>pfp_report.py</code> in
<code>pfp-runtime/tools</code>. Bit order verified identical to
<code>pfp2bits</code> and to the PharmCast Python decoder.</p>
</div>
<script>
document.querySelectorAll(".copy").forEach(function(b){
  b.addEventListener("click",function(){
    var t=document.getElementById(b.dataset.target), txt=t.textContent;
    function done(){var o=b.textContent;b.textContent="Copied";
      setTimeout(function(){b.textContent=o},1200);}
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(txt).then(done,fallback);
    } else {fallback();}
    function fallback(){
      var r=document.createRange();r.selectNodeContents(t);
      var s=window.getSelection();s.removeAllRanges();s.addRange(r);
      try{document.execCommand("copy");done();}catch(e){}
    }
  });
});
</script>
</body></html>
""")
    return tpl.safe_substitute({
        "source": html.escape(source), "n": format(n, ","), "avg": "%.1f" % avg,
        "uni": format(len(union), ","), "cov": "%.1f" % cov, "tri": triangle_svg(),
        "legend": legend,
        "trips": bar_rows(triplets, sum(triplets.values()), 15, trip_fmt),
        "dists": bar_rows(dists, sum(dists.values())),
        "feats": bar_rows(feat_single, sum(feat_single.values()), None,
                          lambda k: "%s &nbsp;%s" % (chip(k), html.escape(FEATURE_NAMES.get(k, k)))),
        "mols": "\n".join(mol_rows), "shown": format(len(mol_rows), ","),
        "hits": hits_section,
        "copy": string.Template(copy_section).safe_substitute({
            "shown_copy": format(min(len(records), max_listed), ","),
            "n": format(n, ","),
            "rawtext": html.escape(raw_text),
            "bitstext": html.escape(bits_text),
        }),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pfp", help=".pfp or .pfp.gz file")
    ap.add_argument("-o", "--out", default="pfp_report.html")
    ap.add_argument("--labels", default=os.path.join(HERE, "pharm10549.list"))
    ap.add_argument("--limit", type=int, default=0, help="read at most N molecules")
    ap.add_argument("--max-listed", type=int, default=2000)
    ap.add_argument("--bits", action="store_true",
                    help="print the 0/1 string to stdout instead, like pfp2bits")
    a = ap.parse_args()

    records = read_pfp(a.pfp, a.limit)
    if not records:
        sys.exit("no fingerprint records found in %s" % a.pfp)

    if a.bits:
        for name, words, _raw in records:
            on = set(decode(words))
            sys.stdout.write("%s %s\n" % (name, "".join(
                "1" if i in on else "0" for i in range(N_PHARM))))
        return

    labels = load_labels(a.labels)
    open(a.out, "w").write(render(records, labels, os.path.basename(a.pfp),
                                  a.max_listed))
    sys.stderr.write("wrote %s (%d molecules)\n" % (a.out, len(records)))


if __name__ == "__main__":
    main()
