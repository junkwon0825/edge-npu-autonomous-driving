"""Move potentially contaminated scenario runs out of a results file (they are kept, not deleted).

The runs are appended to ``<results>_quarantined.json`` with a reason; the resumable runner then
re-executes exactly the missing keys. Their logs are renamed ``*.jsonl.quarantined``.

  python scripts/quarantine_runs.py --scenario lead_vehicle_braking --mode gt_oracle \
      --conditions ideal_0ms gpu_p95 --reason "..."
"""

from __future__ import annotations

import argparse
import json
import time

from npu_ad.config import resolve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/evidence/scenarios_v3.json")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--ego-kmh", type=float, default=30.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reason", required=True)
    a = ap.parse_args()
    res_p = resolve(a.results)
    q_p = res_p.with_name(res_p.stem + "_quarantined.json")
    res = json.loads(res_p.read_text(encoding="utf-8"))
    q = json.loads(q_p.read_text(encoding="utf-8")) if q_p.exists() else {"runs": []}

    def hit(r: dict) -> bool:
        return (
            r["scenario"] == a.scenario
            and r.get("mode") == a.mode
            and r["condition"] in a.conditions
            and float(r.get("ego_kmh", 30.0)) == a.ego_kmh
            and r.get("tag", "") == a.tag
        )

    moved = [r for r in res["runs"] if hit(r)]
    res["runs"] = [r for r in res["runs"] if not hit(r)]
    for r in moved:
        r["quarantine"] = {"reason": a.reason, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        stem = f"{r['scenario']}__{r['condition']}__{r['aeb_version']}__{r['mode']}__{int(r['ego_kmh'])}kmh"
        log = resolve(f"outputs/scenario_logs/{stem}{'__' + a.tag if a.tag else ''}.jsonl")
        if log.exists():
            log.replace(log.with_suffix(".jsonl.quarantined"))
    q["runs"] += moved
    for p, d in ((q_p, q), (res_p, res)):
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        tmp.replace(p)
    print(f"quarantined {len(moved)} runs -> {q_p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
