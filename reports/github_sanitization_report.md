# GitHub sanitization report (before any public push)

Status: **analysis only. No history has been rewritten and nothing has been pushed.**
Scanned on 2026-11-25: all 46 commits on all local branches and tags (`v1.0.0-rc1`, `v2.0.0-rc1`).

In this report the remote container hostname is written as `<REMOTE_HOSTNAME>`, so this file does not
spread it further. The literal replacement list is in `outputs/sanitize/replacements.txt`, which is
git-ignored and stays local.

## Findings
| # | Item | Sensitivity | In HEAD (tracked now) | In history |
|---|---|---|---|---|
| 1 | Remote container hostname `<REMOTE_HOSTNAME>` (contains an organization name) in the `"host"` field of V1 NPU evidence, and in two V1 docs | medium: identifies the provider's machine | **yes, 15 files** (list below) | introduced in `94593a1` (2026-11-22); present in 45 of 46 commit trees, including both tags |
| 2 | Remote home path `<REMOTE_HOME>/npu_auto_driving_p2/...` inside Python tracebacks stored in NPU evidence | low: reveals the remote user is root and the directory name | **yes, 1 file:** `reports/evidence/npu_int8_probe2.json`. `scripts/v2_fetch_npu_status.py` contains `<REMOTE_HOME>/` only as the redaction pattern, which is intended | introduced in `bf872a2`; at most 6 files at `bd8dd8c` (V2 port-lab evidence), reduced to 1 by `bc7aefb` |
| 3 | Author/committer e-mail (a personal Gmail address) in all 46 commits | personal data; public on GitHub after a push | commit metadata | all commits |
| 4 | Local project paths such as `<LOCAL_PROJECT_ROOT>` and `<AGENT_LAB_ROOT>` | low (no user name) | `README.md:334`, `code/04_rgb_camera.py`, `code/05_rgb_sync.py`, `code/06_lidar_sync.py`, `reports/agent_runs/latest_status.md`, `reports/environment_audit.md`, `reports/review_findings.md` | yes |
| 5 | SSH alias `furiosa-npu` and remote directory name `npu_auto_driving_p2` | low: an alias only; host, user and key are not stored | 17 files | yes |

**Correction to an earlier statement.** I previously reported "the remote home path in early V2
commits". That is only partly accurate:
- The `<REMOTE_HOME>/` path was introduced by V1 evidence (`bf872a2`, `6402f5b`).
- V2 evidence briefly added more occurrences (`bd8dd8c`).
- One V1 file in HEAD still contains it.

An earlier scan for `/root` missed all of this. Git Bash rewrites `/root` arguments into Windows
paths; the scan was repeated with `MSYS_NO_PATHCONV=1`.

**Nothing found** in any commit, for:
- private keys (`BEGIN … PRIVATE KEY`)
- GitHub, OpenAI, AWS or Slack token patterns
- `password=` / `password:`
- `IdentityFile`
- SSH user names, `/home/…` paths
- the Windows user name
- the e-mail address inside file contents
- private IP addresses (the only IPv4-like string is a version number)

### Files in HEAD containing `<REMOTE_HOSTNAME>` (finding 1)
- **Evidence JSON:**
  - `reports/evidence/gpu_vs_npu.json`, `gpu_vs_npu_bf16.json`, `gpu_vs_npu_w8.json`
  - `npu_bench.json`, `npu_bench_bf16.json`, `npu_bench_replay.json`, `npu_bench_w8.json`
  - `npu_compile.json`, `npu_compile_bf16.json`, `npu_compile_w8.json`
  - `npu_perf.json`, `npu_perf_bf16.json`, `npu_perf_w8.json`
- **Reports:** `reports/environment_audit.md`, `reports/npu_compilation.md`

Since V2, new evidence records `"host": "<redacted>"` (`scripts/v2_fetch_npu_status.py`).

## Re-scan after V3 (2026-11-25, HEAD after the V3 evidence commits, 56 commits)
- **Unchanged:**
  - `<REMOTE_HOSTNAME>` in the same 15 files
  - `<REMOTE_HOME>/` in `reports/evidence/npu_int8_probe2.json`
- **Intended occurrences:** `<REMOTE_HOME>/` also appears in this report and in the redaction pattern of
  `scripts/v2_fetch_npu_status.py`.
- **New V3 evidence** (`v3_*.json`, closed-loop logs, memory trace): no hostname, no remote path, no
  personal data.
- **Key / token / password patterns:** none found. The only matches are this report's own list of
  search patterns.

## Options
| | A: minimal (current files only) | B: rewrite history | C: clean portfolio repository |
|---|---|---|---|
| What | one new commit replacing the two strings in the 16 current files | `git filter-repo --replace-text` over all commits (+ optional mailmap) | new repository from the validated final tree, with a few curated milestone commits/tags (V1, V2, V3) |
| Hostname / path still public? | **yes, in history** | no | no |
| Personal e-mail in metadata | stays (fixable only with a rewrite) | can be mapped to the GitHub no-reply address | new commits made with the no-reply address |
| Development history | kept | kept (new hashes) | lost; only milestones |
| `code_commit` hashes inside evidence | still resolve | **no longer resolve**: needs the published old → new commit map | do not resolve; needs a note that evidence was produced in a private repository |
| Risk | low; insufficient for a public repo | medium: irreversible without the backup bundle; tags move | low; credibility depends on the reports instead of the commit trail |
| Suitable for | a private repository only | **a public portfolio** | a public portfolio when history is not needed |

**Recommendation: B.**
- The commit trail is part of the engineering evidence: quarantined runs, fixes driven by failed tests,
  evidence committed separately from code. B keeps that trail and removes the two strings.
- Publish the `filter-repo` commit map next to the sanitization log, so `code_commit` fields in the
  evidence can be translated.
- Choose **C** if a smaller, curated history is preferred over the full trail.
- **A** is not enough for a public push.

## Recommended procedure for B (after approval only)
1. Back up: `git bundle create ../carla_project_backup.bundle --all`, then verify it with
   `git bundle verify`.
2. Work on a fresh mirror clone, never in the working repository:
   `git clone --mirror . ../carla_sanitize.git`
3. In the mirror, run
   `git filter-repo --replace-text <path>/replacements.txt`.
   - The replacements are `<REMOTE_HOSTNAME>` → `<REMOTE_HOST>` and `<REMOTE_HOME>/` → `~/`.
   - Optionally add `--mailmap` to use the GitHub no-reply e-mail.
   - `git-filter-repo` is a pip package. Installing it in a project venv is a local install, not a
     system-wide one.
4. Verify in the mirror:
   - `git grep` over all commits returns nothing for both strings
   - tests pass on the rewritten `HEAD`
   - every evidence JSON still parses
5. Tags `v1.0.0-rc1` and `v2.0.0-rc1` are rewritten with new hashes. Record the old → new mapping
   (`filter-repo` writes `commit-map`) in a sanitization log.
6. Push only the sanitized mirror. That needs separate approval.

## Risks of rewriting
- **Every commit hash changes.**
  - Hashes quoted in docs, reports and evidence `code_commit` fields no longer resolve.
  - The commit map is needed to translate them. Hashes written *inside* files are not rewritten.
- **Evidence "frozen" at V1 is modified.**
  - Only a non-measurement metadata field and traceback paths change.
  - This must be documented in the sanitization log, so the evidence is not presented as untouched.
- **Other worktrees or clones** of this repository would diverge and must be re-cloned. Nothing has
  been pushed, so there is no remote copy to reconcile.
- **Irreversible without the backup bundle,** so step 1 is mandatory.

## Items requiring user approval
- [ ] Choose option A, B or C
- [ ] Approve installing `git-filter-repo` in the project venv
- [ ] Approve the history rewrite, on the mirror
- [ ] Decide whether to replace the commit e-mail with the GitHub no-reply address
- [ ] Approve the push, separately
