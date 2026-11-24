"""V3 evidence manifest: size + SHA-256 of every V3 evidence file, table, plot, demo asset and doc (read only).

Also records the model weights used by V3 (hash only; the weights are not committed) and validates that
every evidence JSON parses and every run in the closed-loop files carries a code commit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOBS = ["reports/evidence/v3_*", "reports/v3_*.md", "assets/plots/v3_*.png", "assets/demo/v3/*",
         "assets/diagrams/v3_*.png", "docs/v3_*.md", "docs/npu_*.md", "docs/portfolio_key_results.md",
         "docs/interview_questions_v3.md", "reports/github_sanitization_report.md", "README.md"]  # fmt: skip
WEIGHTS = ["models/yolo26n.pt", "models/lane_unet_s3c32.pt"]
CLOSED_LOOP = ["v3_closed_loop.json", "v3_closed_loop_rerun_nightrain.json", "v3_closed_loop_rerun_stale.json"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = sorted({p for g in GLOBS for p in ROOT.glob(g) if p.is_file()})
    rows = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    problems = []
    for p in files:
        if p.suffix == ".json":
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except ValueError as e:
                problems.append(f"{p.name}: {e}")
    runs = {}
    for name in CLOSED_LOOP:
        d = json.loads((ROOT / "reports" / "evidence" / name).read_text(encoding="utf-8"))
        runs[name] = {"runs": len(d["runs"]), "code_commits": sorted({r["code_commit"] for r in d["runs"]})}
        problems += [f"{name}: run without code_commit" for r in d["runs"] if not r.get("code_commit")]
    weights = [{"path": w, "sha256": sha256(ROOT / w), "committed": False} for w in WEIGHTS if (ROOT / w).exists()]
    out = ROOT / "reports" / "v3" / "evidence_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"release": "v3.0.0-rc1", "closed_loop": runs, "weights": weights, "validation_problems": problems,
           "files": rows}  # fmt: skip
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(len(rows), "files,", runs, "problems:", problems, "->", out)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
