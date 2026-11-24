"""V2 evidence manifest: size + SHA-256 of every V2 evidence file, plot, demo asset and doc (read only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOBS = ["reports/evidence/v2_*.json", "assets/plots/v2_*.png", "assets/demo/v2_*", "assets/diagrams/v2_*.png",
         "docs/v2_*.md", "docs/npu_operator_compatibility.md", "reports/v2_*.md", "configs/*.json"]  # fmt: skip


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted({p for g in GLOBS for p in ROOT.glob(g) if p.is_file()})
    out = ROOT / "reports" / "v2" / "evidence_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    out.write_text(json.dumps({"release": "v2.0.0-rc1", "files": rows}, indent=2), encoding="utf-8")
    print(len(rows), "files ->", out)


if __name__ == "__main__":
    main()
