"""Freeze the V1 release evidence without copying or modifying it.

Writes ``reports/v1/evidence_manifest.json`` (size + SHA-256 of every validated evidence file, plot,
config and demo asset) and ``reports/v1/aeb_summary.json`` (per-mode counts recomputed from
``reports/evidence/scenarios_v3.json``). Copies the small demo ``summary.json`` from the git-ignored
``outputs/v1/demo`` into ``reports/v1/demo_summary.json``. Source evidence files are only read.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "reports" / "v1"
GLOBS = [
    "reports/evidence/*.json",
    "assets/plots/*.png",
    "assets/demo/*",
    "assets/diagrams/*.png",
    "configs/*.json",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aeb_summary(runs: list[dict], kmh: float = 30.0) -> dict:
    out: dict = {"source": "reports/evidence/scenarios_v3.json", "ego_kmh": kmh, "modes": {}}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        if float(r.get("ego_kmh", 30.0)) != kmh or r.get("tag"):
            continue  # speed sweep and repeat runs are reported separately
        key = "aeb_off" if r["condition"] == "aeb_off" else r.get("mode", "gt_gated")
        groups[key].append(r)
    for key, rs in sorted(groups.items()):
        haz = [r for r in rs if r["hazard"]]
        safe = [r for r in rs if not r["hazard"]]
        gaps = [r["min_gap_m"] for r in haz if r.get("min_gap_m") is not None]
        ttc = [r["gt_ttc_at_first_brake_s"] for r in haz if r.get("gt_ttc_at_first_brake_s") is not None]
        out["modes"][key] = {
            "hazard_runs": len(haz),
            "hazard_collisions": sum(bool(r["collision"]) for r in haz),
            "no_hazard_runs": len(safe),
            "no_hazard_collisions": sum(bool(r["collision"]) for r in safe),
            "false_positive_brake_events": sum(int(r.get("false_positive_brake_events") or 0) for r in safe),
            "min_gap_m": {"min": min(gaps), "median": statistics.median(gaps)} if gaps else None,
            "gt_ttc_at_first_brake_s_median": statistics.median(ttc) if ttc else None,
            "conditions": sorted({r["condition"] for r in rs}),
        }
    return out


def main() -> None:
    V1.mkdir(parents=True, exist_ok=True)
    files = sorted({p for g in GLOBS for p in ROOT.glob(g) if p.is_file()})
    manifest = {
        "release": "v1.0.0-rc1",
        "note": "validated evidence referenced in place; nothing copied or overwritten",
        "files": [
            {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files
        ],
    }
    (V1 / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runs = json.loads((ROOT / "reports/evidence/scenarios_v3.json").read_text(encoding="utf-8"))["runs"]
    (V1 / "aeb_summary.json").write_text(json.dumps(aeb_summary(runs), indent=2), encoding="utf-8")
    demo = ROOT / "outputs/v1/demo/summary.json"
    if demo.exists():
        (V1 / "demo_summary.json").write_text(demo.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(files)} files in manifest -> {V1}")


if __name__ == "__main__":
    main()
