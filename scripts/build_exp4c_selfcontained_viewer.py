#!/usr/bin/env python3
"""Build fully self-contained Exp 4c HTML viewers (no external GLB files).

Embeds downsampled surface point clouds from view_gt.glb / view_assembly_0.glb
directly in each HTML file (Three.js inlined). Safe to download and open locally.

Usage:
  python scripts/build_exp4c_selfcontained_viewer.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_viz import THREEJS_CDN, ORBIT_CDN, load_asset  # noqa: E402

OUT = ROOT / "viz_output" / "exp4c_thinwall"
PTS_PER_PART = 600
COLORS = [
    "#4ade80", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa",
    "#fb923c", "#2dd4bf", "#f87171", "#94a3b8", "#e879f9",
]


def fps_downsample(pts: np.ndarray, k: int) -> np.ndarray:
    if pts.shape[0] <= k:
        return pts
    selected = [0]
    dists = np.full(pts.shape[0], np.inf)
    for _ in range(k - 1):
        d = np.linalg.norm(pts - pts[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return pts[selected]


def sample_glb_parts(glb_path: Path, pts_per_part: int) -> list[list[float]]:
    """Return list of point clouds (world coords), one per mesh in the GLB."""
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        meshes = loaded.dump(concatenate=False)
    parts: list[np.ndarray] = []
    for mesh in meshes:
        if len(getattr(mesh, "faces", [])) == 0:
            continue
        try:
            pts, _ = trimesh.sample.sample_surface_even(
                mesh, max(pts_per_part * 4, 400)
            )
        except Exception:
            pts, _ = trimesh.sample.sample_surface(mesh, max(pts_per_part * 4, 400))
        pts = fps_downsample(np.asarray(pts, dtype=np.float64), pts_per_part)
        parts.append(pts)
    if not parts:
        raise ValueError(f"No geometry in {glb_path}")
    return [p.tolist() for p in parts]


def find_metrics(thinwall_stamp: str) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    diag = ROOT / "logs" / "diagnostics"
    for run_dir in sorted(diag.glob(f"thinwall_{thinwall_stamp}_*_init1_s41")):
        for jpath in sorted((run_dir / "version_0" / "json_results").glob("*.json")):
            j = json.loads(jpath.read_text())
            metrics[j["name"]] = j
    return metrics


def collect_samples(thinviz_stamp: str) -> list[dict]:
    samples = []
    deploy = ROOT / "logs" / "deploy"
    for cat_dir in sorted(deploy.glob(f"thinviz_{thinviz_stamp}_*")):
        cat = cat_dir.name.split(f"thinviz_{thinviz_stamp}_", 1)[-1]
        asm_root = cat_dir / "version_0" / "assembly_results" / cat
        if not asm_root.is_dir():
            continue
        for sample_dir in sorted(asm_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            gt = sample_dir / "view_gt.glb"
            pred = sample_dir / "view_assembly_0.glb"
            if gt.exists() and pred.exists():
                samples.append({
                    "category": cat,
                    "sample": sample_dir.name,
                    "key": f"{cat}/{sample_dir.name}",
                    "gt": gt,
                    "pred": pred,
                })
    return samples


def acc_class(v: float) -> str:
    if v >= 0.8:
        return "good"
    if v >= 0.4:
        return "warn"
    return "bad"


def viewer_html(
    key: str,
    metrics: dict,
    gt_parts: list[list[list[float]]],
    pred_parts: list[list[list[float]]],
    colors: list[str],
    three_js: str,
    orbit_js: str,
) -> str:
    pa = float(metrics.get("part_acc", 0)) * 100
    rr = float(metrics.get("rmse_r", 0))
    rt = float(metrics.get("rmse_t", 0))
    cd = float(metrics.get("shape_cd", 0))
    nparts = len(gt_parts)
    data_json = json.dumps(
        {"gt": gt_parts, "pred": pred_parts, "colors": colors[:nparts]},
        separators=(",", ":"),
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>GARF Exp4c — {key}</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#0a0e1a;color:#e0e0e0;font-family:system-ui,sans-serif;overflow:hidden}}
    #c{{position:absolute;inset:0}}
    #ui{{
      position:absolute;top:16px;right:16px;width:280px;padding:16px;border-radius:10px;
      background:rgba(10,14,26,.95);border:1px solid #1e293b;z-index:10;display:flex;flex-direction:column;gap:10px
    }}
    h1{{font-size:13px;letter-spacing:2px;text-transform:uppercase;color:#7dd3fc}}
    .sub{{font-size:10px;color:#64748b;word-break:break-all}}
    .metric-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
    .metric-card{{padding:8px;border-radius:6px;background:#0f172a;border:1px solid #1e293b;text-align:center}}
    .metric-card .val{{font-size:16px;font-weight:700}}
    .metric-card .lbl{{font-size:9px;color:#64748b;text-transform:uppercase;margin-top:2px}}
    .good .val{{color:#4ade80}}.warn .val{{color:#fbbf24}}.bad .val{{color:#f87171}}
    .mode-btn{{
      padding:8px;border-radius:5px;border:1px solid #1e293b;background:#020617;
      color:#94a3b8;font-size:10px;text-transform:uppercase;cursor:pointer
    }}
    .mode-btn.active{{border-color:#0ea5e9;background:#0c2744;color:#e0f2fe}}
    #mode-buttons{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
    #mode-label{{
      position:absolute;top:16px;left:16px;padding:6px 14px;border-radius:999px;
      border:1px solid #1e293b;background:rgba(10,14,26,.95);font-size:10px;
      letter-spacing:2px;text-transform:uppercase;color:#7dd3fc;z-index:10
    }}
    a{{color:#7dd3fc;font-size:11px;text-decoration:none}}
    #hint{{font-size:10px;color:#475569;text-align:center}}
  </style>
</head>
<body>
  <div id="c"></div>
  <div id="mode-label">Ground Truth</div>
  <div id="ui">
    <div><h1>Exp 4c (self-contained)</h1><div class="sub">{key}</div></div>
    <div class="metric-grid">
      <div class="metric-card {acc_class(metrics.get('part_acc',0))}"><div class="val">{pa:.1f}%</div><div class="lbl">Part Acc</div></div>
      <div class="metric-card"><div class="val">{rr:.1f}°</div><div class="lbl">RMSE R</div></div>
      <div class="metric-card"><div class="val">{rt:.3f}</div><div class="lbl">RMSE T</div></div>
      <div class="metric-card"><div class="val">{cd:.4f}</div><div class="lbl">Shape CD</div></div>
    </div>
    <div id="mode-buttons">
      <button class="mode-btn active" id="btn-gt">Ground Truth</button>
      <button class="mode-btn" id="btn-pred">GARF Predicted</button>
    </div>
    <a href="index.html">← Back to index</a>
    <div id="hint">Self-contained · drag orbit · scroll zoom</div>
  </div>
  <script>{three_js}</script>
  <script>{orbit_js}</script>
  <script>
    const DATA = {data_json};
    let scene, camera, renderer, controls;
    let gtGroup, predGroup;
    let mode = 'gt';

    function buildGroup(parts) {{
      const g = new THREE.Group();
      for (let i = 0; i < parts.length; i++) {{
        const pts = parts[i];
        const pos = new Float32Array(pts.length * 3);
        for (let j = 0; j < pts.length; j++) {{
          pos[3*j]=pts[j][0]; pos[3*j+1]=pts[j][1]; pos[3*j+2]=pts[j][2];
        }}
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({{
          size: 0.012, color: new THREE.Color(DATA.colors[i % DATA.colors.length]),
          sizeAttenuation: true
        }});
        g.add(new THREE.Points(geom, mat));
      }}
      return g;
    }}

    function setMode(m) {{
      mode = m;
      gtGroup.visible = (m === 'gt');
      predGroup.visible = (m === 'pred');
      document.getElementById('mode-label').textContent = m === 'gt' ? 'Ground Truth' : 'GARF Predicted';
      document.getElementById('btn-gt').classList.toggle('active', m === 'gt');
      document.getElementById('btn-pred').classList.toggle('active', m === 'pred');
    }}

    function init() {{
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0e1a);
      camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.001, 100);
      renderer = new THREE.WebGLRenderer({{antialias:true}});
      renderer.setSize(innerWidth, innerHeight);
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      document.getElementById('c').appendChild(renderer.domElement);
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      const d = new THREE.DirectionalLight(0xffffff, 0.8); d.position.set(2,3,4); scene.add(d);
      gtGroup = buildGroup(DATA.gt);
      predGroup = buildGroup(DATA.pred);
      predGroup.visible = false;
      scene.add(gtGroup); scene.add(predGroup);
      const box = new THREE.Box3().setFromObject(gtGroup);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      camera.position.set(maxDim*1.2, maxDim*0.9, maxDim*1.4);
      controls.target.set(0,0,0);
      document.getElementById('btn-gt').onclick = () => setMode('gt');
      document.getElementById('btn-pred').onclick = () => setMode('pred');
      window.addEventListener('resize', () => {{
        camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
      }});
      (function anim(){{ requestAnimationFrame(anim); controls.update(); renderer.render(scene,camera); }})();
    }}
    init();
  </script>
</body>
</html>"""


def index_html(rows: list[dict]) -> str:
    mean_pa = sum(r["part_acc"] for r in rows) / len(rows) * 100 if rows else 0
    body = []
    for i, r in enumerate(rows):
        pa = r["part_acc"] * 100
        body.append(f"""
        <tr onclick="window.open('{r['html']}','_blank')" style="cursor:pointer">
          <td>{i}</td><td>{r['category']}</td><td>{r['sample']}</td><td>{r['num_parts']}</td>
          <td class="{acc_class(r['part_acc'])}">{pa:.1f}%</td>
          <td>{r['rmse_r']:.1f}°</td><td>{r['rmse_t']:.3f}</td><td>{r['shape_cd']:.4f}</td>
        </tr>""")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>GARF Exp 4c — Thin-wall (self-contained)</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0a0e1a;color:#e0e0e0;font-family:system-ui,sans-serif;padding:32px}}
  h1{{font-size:22px;letter-spacing:3px;text-transform:uppercase;color:#7dd3fc;margin-bottom:8px}}
  .sub{{color:#64748b;font-size:12px;margin-bottom:20px;max-width:820px;line-height:1.5}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th,td{{padding:10px 12px;border-bottom:1px solid #1e293b;text-align:left}}
  th{{color:#64748b;font-size:10px;text-transform:uppercase}}
  tr:hover{{background:#0f172a;cursor:pointer}}
  .good{{color:#4ade80}}.warn{{color:#fbbf24}}.bad{{color:#f87171}}
</style></head><body>
<h1>Exp 4c — Real Thin-wall (self-contained)</h1>
<div class="sub">
  Each viewer HTML embeds the mesh point cloud — no GLB files required.
  Download any <code>.html</code> file and open offline. Ceramics (pottery) assemble well;
  egg/bones shown for comparison.
</div>
<table><thead><tr><th>#</th><th>Cat</th><th>Sample</th><th>Parts</th>
<th>Part Acc</th><th>RMSE R</th><th>RMSE T</th><th>Shape CD</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p style="margin-top:16px;color:#64748b;font-size:11px">Mean part acc: {mean_pa:.1f}% · {len(rows)} samples</p>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinviz-stamp", default="20260605_163400")
    ap.add_argument("--thinwall-stamp", default="20260605_162150")
    ap.add_argument("--pts-per-part", type=int, default=PTS_PER_PART)
    ap.add_argument("--categories", default="", help="Comma list, e.g. ceramics,egg (default: all)")
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    metrics_map = find_metrics(args.thinwall_stamp)
    samples = collect_samples(args.thinviz_stamp)
    if cats:
        samples = [s for s in samples if s["category"] in cats]

    OUT.mkdir(parents=True, exist_ok=True)
    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    print("Embedding Three.js…")
    three_js = load_asset(str(assets), "three.min.js", THREEJS_CDN)
    orbit_js = load_asset(str(assets), "OrbitControls.js", ORBIT_CDN)

    rows = []
    for s in samples:
        print(f"  sampling {s['key']}…", flush=True)
        gt_raw = sample_glb_parts(s["gt"], args.pts_per_part)
        pred_raw = sample_glb_parts(s["pred"], args.pts_per_part)
        all_gt = np.vstack([np.asarray(p) for p in gt_raw])
        center = all_gt.mean(axis=0)

        def apply_center(raw: list[list[list[float]]]) -> list[list[list[float]]]:
            out = []
            for part in raw:
                arr = np.asarray(part, dtype=np.float64) - center
                out.append(arr.tolist())
            return out

        gt_c = apply_center(gt_raw)
        pred_c = apply_center(pred_raw)
        m = metrics_map.get(s["key"], {})
        html_name = f"{s['category']}_{s['sample']}.html"
        html = viewer_html(s["key"], m, gt_c, pred_c, COLORS, three_js, orbit_js)
        (OUT / html_name).write_text(html)
        sz = (OUT / html_name).stat().st_size / 1024 / 1024
        print(f"    -> {html_name} ({sz:.1f} MB)")
        rows.append({
            "category": s["category"], "sample": s["sample"], "html": html_name,
            "num_parts": len(gt_c),
            "part_acc": float(m.get("part_acc", 0)),
            "rmse_r": float(m.get("rmse_r", 0)),
            "rmse_t": float(m.get("rmse_t", 0)),
            "shape_cd": float(m.get("shape_cd", 0)),
        })

    (OUT / "index.html").write_text(index_html(rows))
    print(f"\nDone: {OUT}/index.html + {len(rows)} self-contained viewers")
    print("Download index.html + any sample .html — no other files needed.")


if __name__ == "__main__":
    main()
