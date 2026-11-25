# GitHub sanitization log

Status: **history rewritten in a separate sanitized clone (option B), verified, and published from that clone.**
Date: 2026-11-25. Tool: `git-filter-repo` 2.47.0, installed in the project virtual environment only.

## What was done
1. **Backup.** A full bundle of the private repository (`git bundle create … --all`) was made and checked
   with `git bundle verify`. Refs, commit list, HEAD tree and working-tree status were recorded before the
   rewrite. The backup stays outside this repository.
2. **Separate clone.** The rewrite ran in a fresh clone. The private development repository was not
   modified.
3. **Rewrite:** `git filter-repo --replace-text <rules> --replace-message <rules> --mailmap <mailmap>`
   over all branches and tags. The rule file is kept locally, because it contains the original strings.
4. **Tags.** `v1.0.0-rc1` and `v2.0.0-rc1` were rewritten in place: same names, messages and tagger
   timestamps. The old `v3.0.0-rc1` was dropped. It is re-created on the final sanitized V3 commit, after
   the public-facing checks.

**Rewritten:** 59 commits on 4 branches. Author names, commit messages, commit order and author/committer
timestamps are unchanged.

| Replaced (all blobs and messages, all commits) | Replacement | Where it occurred |
|---|---|---|
| remote NPU container hostname (contains an organization name) | `<REMOTE_NPU_HOST>` | `"host"` field of 13 V1 NPU evidence files, 2 V1 reports |
| the organization name on its own (case-insensitive safeguard) | `<ORG>` | no further occurrences |
| remote home directory (`/` + `root/`) in compiler cache paths and tracebacks | `<REMOTE_HOME>/` | V1 and V2 NPU evidence (port lab, INT8 probe, perf tracebacks) |
| local project root on the development PC | `<LOCAL_PROJECT_ROOT>` | README, early reports, 3 CARLA learning scripts |
| local paths of the AI-agent tooling used during development | `<AGENT_LAB_ROOT>` | early process logs |
| local virtual-environment directory | `<LOCAL_VENV_DIR>` | environment audit |
| personal author/committer/tagger e-mail | the owner's GitHub no-reply address | all commits and tags |

**Where the no-reply address comes from.** It was taken from the owner's GitHub account (`gh api user`
gives the account id), and the owner's GitHub-generated merge commits in another repository already use
it. It was not constructed by hand.

**Evidence is modified only in metadata.** The rewrite changed only:
- a non-measurement `"host"` field
- path prefixes inside error tracebacks and compiler cache messages

No measured value was changed. The V1/V2 SHA-256 manifests (`reports/v1/`, `reports/v2/`) are kept
unchanged and describe the files as they were before sanitization. `scripts/verify_manifests.py` checks
them against this checkout. It accepts CRLF or LF, because the original manifests were generated on a
CRLF Windows checkout.

| Manifest | Files | Match | Differ | Why they differ |
|---|---|---|---|---|
| V1 | 46 | 32 | 14 | exactly the 14 sanitized NPU evidence files (`host` field, traceback paths) |
| V2 | 49 | 48 | 1 | `docs/npu_operator_compatibility.md` was regenerated in V3 (74 rows), after the V2 manifest; it matches the V3 manifest |
| V3 | regenerated after sanitization | all | 0 | — |

## Commit hashes
- **All hashes changed.** Hashes written inside files were not rewritten. This includes `code_commit` in
  evidence JSON and hashes quoted in reports.
- **Translation tables:**
  - [`sanitization/commit_map.txt`](sanitization/commit_map.txt): old → new, for every commit
  - [`sanitization/ref_map.txt`](sanitization/ref_map.txt): branches and tags
- **Example:** the V3 matrix evidence cites `5400927` and `75d8ecd`. In this repository they are
  `265fc5e` and `7d0874d`.

## Post-rewrite scan (all reachable commits and tags, text and binary blobs)
| Check | Result |
|---|---|
| remote hostname, organization name | 0 |
| personal e-mail (contents, author, committer, tagger) | 0 |
| Windows user name, user-profile paths, `/home/<user>` | 0 |
| local project / tooling / venv absolute paths | 0 (only the placeholders remain) |
| private keys, GitHub / OpenAI / Anthropic / AWS / Slack / Google token patterns | 0 |
| password / secret / token assignments | 0 |
| `.env`, `*.pem`, `*.key`, `id_rsa*` files ever committed | 0 |
| model weights, compiler caches, replay data, `outputs/`, `data/` ever committed | 0 |

**Allowed occurrences** (reviewed; not private):
- **Redaction code:** `scripts/v2_fetch_npu_status.py` contains the remote-home redaction pattern
  itself.
- **Install path:** `C:\CARLA_0.9.16` is the CARLA installation directory, with no user name.
  - It appears in V1/V3 reports and in the `carla` wheel line of `requirements-lock-windows.txt`.
- **IPv4-like strings:** the only one is a package version number.
- **SSH alias and remote directory name:** `furiosa-npu` and `npu_auto_driving_p2`. They are an alias
  and a directory name; host, user and key are not stored.

## Size
- **Repository size:** pack 22.4 MiB, about 310 tracked files.
- **Largest files:** the four demo videos, 3.1–5.5 MB each. The full-size V3 video stays local.
- **Not tracked:** weights (`*.pt`, `*.onnx`), compiler caches, datasets, replay frames and local outputs
  are ignored by `.gitignore`.
