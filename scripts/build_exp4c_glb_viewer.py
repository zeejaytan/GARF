#!/usr/bin/env python3
"""Build interactive HTML viewer for Exp 4c thin-wall GLB assemblies.

Reads GARF-exported GLBs from logs/deploy/thinviz_* and benchmark metrics from
logs/diagnostics/thinwall_* (seed 41). Writes self-contained-ish HTML to
viz_output/exp4c_thinwall/ with symlinks to GLBs (no copy — ~1.5 GB).

Usage:
  python scripts/build_exp4c_glb_viewer.py \\
      --thinviz-stamp 20260605_163400 \\
      --thinwall-stamp 20260605_162150
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viz_output" / "exp4c_thinwall"
THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"
ORBIT_CDN = "https://unpkg.com/three@0.134.0/examples/js/controls/OrbitControls.js"
GLTF_CDN = "https://unpkg.com/three@0.134.0/examples/js/loaders/GLTFLoader.js"


def load_asset(out_dir: Path, name: str, url: str) -> str:
    path = out_dir / name
    if not path.exists():
        import urllib.request
        print(f"  download {name}")
        path.write_text(urllib.request.urlopen(url, timeout=60).read().decode("utf-8"))
    return path.read_text()


def find_metrics(thinwall_stamp: str) -> dict[str, dict]:
    """Map 'category/sample' -> metrics from seed-41 JSON."""
    metrics: dict[str, dict] = {}
    diag = ROOT / "logs" / "diagnostics"
    for run_dir in sorted(diag.glob(f"thinwall_{thinwall_stamp}_*_init1_s41")):
        m = re.match(rf"thinwall_{thinwall_stamp}_(?P<cat>\w+)_init1_s41", run_dir.name)
        if not m:
            continue
        cat = m.group("cat")
        for jpath in sorted((run_dir / "version_0" / "json_results").glob("*.json")):
            j = json.loads(jpath.read_text())
            name = j.get("name", "")
            # name like ceramics/narrow_bottle4
            key = name if "/" in name else f"{cat}/{name}"
            metrics[key] = {
                "num_parts": j.get("num_parts"),
                "part_acc": j.get("part_acc", 0),
                "rmse_r": j.get("rmse_r", 0),
                "rmse_t": j.get("rmse_t", 0),
                "shape_cd": j.get("shape_cd", 0),
            }
    return metrics


def collect_glb_samples(thinviz_stamp: str) -> list[dict]:
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
                    "gt_path": gt,
                    "pred_path": pred,
                })
    return samples


def symlink_glbs(samples: list[dict], glb_root: Path) -> None:
    glb_root.mkdir(parents=True, exist_ok=True)
    for s in samples:
        dest = glb_root / s["category"] / s["sample"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(s["gt_path"].parent.resolve())


def acc_class(v: float) -> str:
    if v >= 0.8:
        return "good"
    if v >= 0.4:
        return "warn"
    return "bad"


def viewer_html(
    key: str,
    metrics: dict,
    gt_rel: str,
    pred_rel: str,
    three_js: str,
    orbit_js: str,
    gltf_js: str,
) -> str:
    pa = metrics.get("part_acc", 0) * 100
    rr = metrics.get("rmse_r", 0)
    rt = metrics.get("rmse_t", 0)
    cd = metrics.get("shape_cd", 0)
    nparts = metrics.get("num_parts", "?")
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
      background:rgba(10,14,26,.95);border:1px solid #1e293b;z-index:10;
      display:flex;flex-direction:column;gap:10px
    }}
    h1{{font-size:13px;letter-spacing:2px;text-transform:uppercase;color:#7dd3fc}}
    .sub{{font-size:10px;color:#64748b}}
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
    a:hover{{text-decoration:underline}}
    #hint{{font-size:10px;color:#475569;text-align:center}}
    #loading{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
      background:#0a0e1a;color:#64748b;font-size:13px;z-index:5}}
  </style>
</head>
<body>
  <div id="loading">Loading GLB…</div>
  <div id="c"></div>
  <div id="mode-label">Ground Truth</div>
  <div id="ui">
    <div><h1>Exp 4c Thin-wall</h1><div class="sub">{key} · {nparts} parts</div></div>
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
    <a href="../index.html">← Back to index</a>
    <div id="hint">Drag orbit · scroll zoom · right-drag pan</div>
  </div>
  <script>{three_js}</script>
  <script>{orbit_js}</script>
  <script>{gltf_js}</script>
  <script>
    const GT_URL = "{gt_rel}";
    const PRED_URL = "{pred_rel}";
    let scene, camera, renderer, controls, currentModel = null;
    let mode = 'gt';

    function init() {{
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0e1a);
      camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.001, 10000);
      renderer = new THREE.WebGLRenderer({{antialias:true}});
      renderer.setSize(innerWidth, innerHeight);
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      document.getElementById('c').appendChild(renderer.domElement);
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const d1 = new THREE.DirectionalLight(0xffffff, 0.85); d1.position.set(2,3,4); scene.add(d1);
      const d2 = new THREE.DirectionalLight(0x7dd3fc, 0.35); d2.position.set(-3,-1,-2); scene.add(d2);
      window.addEventListener('resize', () => {{
        camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
      }});
      document.getElementById('btn-gt').onclick = () => setMode('gt');
      document.getElementById('btn-pred').onclick = () => setMode('pred');
      loadModel(GT_URL);
      animate();
    }}

    function setMode(m) {{
      mode = m;
      document.getElementById('mode-label').textContent = m === 'gt' ? 'Ground Truth' : 'GARF Predicted';
      document.getElementById('btn-gt').classList.toggle('active', m === 'gt');
      document.getElementById('btn-pred').classList.toggle('active', m === 'pred');
      loadModel(m === 'gt' ? GT_URL : PRED_URL);
    }}

    function loadModel(url) {{
      document.getElementById('loading').style.display = 'flex';
      if (currentModel) {{ scene.remove(currentModel); currentModel = null; }}
      const loader = new THREE.GLTFLoader();
      loader.load(url, (gltf) => {{
        currentModel = gltf.scene;
        scene.add(currentModel);
        const box = new THREE.Box3().setFromObject(currentModel);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        currentModel.position.sub(center);
        camera.position.set(maxDim*1.4, maxDim*1.0, maxDim*1.6);
        controls.target.set(0,0,0);
        controls.update();
        document.getElementById('loading').style.display = 'none';
      }}, undefined, (err) => {{
        document.getElementById('loading').textContent = 'Failed to load GLB: ' + err;
      }});
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    init();
  </script>
</body>
</html>"""


def index_html(rows: list[dict]) -> str:
    mean_pa = sum(r["part_acc"] for r in rows) / len(rows) * 100 if rows else 0
    mean_rr = sum(r["rmse_r"] for r in rows) / len(rows) if rows else 0
    body_rows = []
    for i, r in enumerate(rows):
        pa = r["part_acc"] * 100
        cls = acc_class(r["part_acc"])
        body_rows.append(f"""
        <tr onclick="window.open('viewers/{r['viewer']}.html','_blank')" style="cursor:pointer">
          <td>{i}</td>
          <td>{r['category']}</td>
          <td title="{r['key']}">{r['sample']}</td>
          <td>{r['num_parts']}</td>
          <td class="{cls}">{pa:.1f}%</td>
          <td>{r['rmse_r']:.1f}°</td>
          <td>{r['rmse_t']:.3f}</td>
          <td>{r['shape_cd']:.4f}</td>
        </tr>""")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>GARF Exp 4c — Thin-wall Real Data</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#0a0e1a;color:#e0e0e0;font-family:system-ui,sans-serif;padding:32px}}
    h1{{font-size:22px;letter-spacing:3px;text-transform:uppercase;color:#7dd3fc;margin-bottom:4px}}
    .sub{{color:#64748b;font-size:12px;margin-bottom:24px;max-width:800px;line-height:1.5}}
    .summary{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
    .summary-card{{padding:16px 24px;border-radius:8px;background:#0f172a;border:1px solid #1e293b;text-align:center}}
    .summary-card .val{{font-size:24px;font-weight:700;color:#e2e8f0}}
    .summary-card .lbl{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin-top:4px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th{{text-align:left;padding:10px 12px;border-bottom:2px solid #1e293b;color:#64748b;font-size:10px;text-transform:uppercase}}
    td{{padding:10px 12px;border-bottom:1px solid #1e293b}}
    tr:hover{{background:#0f172a}}
    .good{{color:#4ade80}}.warn{{color:#fbbf24}}.bad{{color:#f87171}}
  </style>
</head>
<body>
  <h1>Exp 4c — Real Thin-wall Objects</h1>
  <div class="sub">
    Fractura_real benchmark: egg (ultra-thin), ceramics (pottery ≈ Juglet), bones (control).
    Click a row to open the 3D GLB viewer (toggle GT vs GARF predicted assembly).
    Ceramics assemble well → thin-wall alone is not the Juglet cause.
  </div>
  <div class="summary">
    <div class="summary-card"><div class="val">{mean_pa:.1f}%</div><div class="lbl">Mean Part Acc</div></div>
    <div class="summary-card"><div class="val">{mean_rr:.1f}°</div><div class="lbl">Mean RMSE Rotation</div></div>
    <div class="summary-card"><div class="val">{len(rows)}</div><div class="lbl">Samples</div></div>
  </div>
  <table>
    <thead><tr>
      <th>#</th><th>Category</th><th>Sample</th><th>Parts</th>
      <th>Part Acc</th><th>RMSE(R)</th><th>RMSE(T)</th><th>Shape CD</th>
    </tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thinviz-stamp", default="20260605_163400")
    ap.add_argument("--thinwall-stamp", default="20260605_162150")
    args = ap.parse_args()

    metrics_map = find_metrics(args.thinwall_stamp)
    samples = collect_glb_samples(args.thinviz_stamp)
    if not samples:
        raise SystemExit("No GLB samples found — check thinviz stamp")

    OUT.mkdir(parents=True, exist_ok=True)
    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    print("Loading Three.js assets…")
    three_js = load_asset(assets, "three.min.js", THREE_CDN)
    orbit_js = load_asset(assets, "OrbitControls.js", ORBIT_CDN)
    gltf_js = load_asset(assets, "GLTFLoader.js", GLTF_CDN)

    glb_root = OUT / "glbs"
    print(f"Symlinking {len(samples)} sample GLB dirs → {glb_root}")
    symlink_glbs(samples, glb_root)

    viewers_dir = OUT / "viewers"
    viewers_dir.mkdir(exist_ok=True)
    table_rows = []

    for s in samples:
        m = metrics_map.get(s["key"], {})
        viewer_name = f"{s['category']}_{s['sample']}"
        gt_rel = f"../glbs/{s['category']}/{s['sample']}/view_gt.glb"
        pred_rel = f"../glbs/{s['category']}/{s['sample']}/view_assembly_0.glb"
        html = viewer_html(s["key"], m, gt_rel, pred_rel, three_js, orbit_js, gltf_js)
        (viewers_dir / f"{viewer_name}.html").write_text(html)
        table_rows.append({
            "category": s["category"],
            "sample": s["sample"],
            "key": s["key"],
            "viewer": viewer_name,
            "num_parts": m.get("num_parts", "?"),
            "part_acc": m.get("part_acc", 0),
            "rmse_r": m.get("rmse_r", 0),
            "rmse_t": m.get("rmse_t", 0),
            "shape_cd": m.get("shape_cd", 0),
        })
        print(f"  {s['key']}  part_acc={m.get('part_acc', 'n/a')}")

    (OUT / "index.html").write_text(index_html(table_rows))
    print(f"\nWrote {OUT}/index.html")
    print(f"     {len(table_rows)} viewers in {viewers_dir}/")
    print("Open index.html in a browser (file:// or via a local HTTP server).")


if __name__ == "__main__":
    main()
