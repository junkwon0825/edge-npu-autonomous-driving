"""Verify the V1/V2/V3 evidence manifests against the files in this checkout (read only).

The V1 and V2 manifests were generated on a Windows checkout whose text files had CRLF line endings,
while Git stores them with LF (`.gitattributes`: `eol=lf`). A row is therefore accepted if the file
matches its SHA-256 either as checked out or with LF converted to CRLF. Rows that match neither are
listed. After the history sanitization these are exactly the files whose host field or traceback paths
were replaced (see `reports/github_sanitization_report.md`).

  python scripts/verify_manifests.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [f"reports/{v}/evidence_manifest.json" for v in ("v1", "v2", "v3")]


def digests(p: Path) -> set[str]:
    data = p.read_bytes()
    out = {hashlib.sha256(data).hexdigest()}
    if b"\0" not in data[:8000] and b"\r\n" not in data:
        out.add(hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest())
    return out


def main() -> int:
    status = 0
    for m in MANIFESTS:
        doc = json.loads((ROOT / m).read_text(encoding="utf-8"))
        rows = doc.get("files", [])
        missing = [r["path"] for r in rows if not (ROOT / r["path"]).exists()]
        changed = [r["path"] for r in rows if r["path"] not in missing and r["sha256"] not in digests(ROOT / r["path"])]
        print(f"{m}: {len(rows)} files, {len(rows) - len(missing) - len(changed)} match, "
              f"{len(changed)} changed, {len(missing)} missing")  # fmt: skip
        for p in changed:
            print("  changed:", p)
        for p in missing:
            print("  missing:", p)
        status |= bool(missing)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
