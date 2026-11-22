# Independent Review (Agent 3 — Kiro, read-only) and Coordinator Verification

- Orchestrator: `agent_lab.cli_agents.workflow_cli` (Phase B) in `<AGENT_LAB_ROOT>`,
  run `wf-20261122-113709-3b1053`, one read-only task, agent `kiro`, 36.7 s, exit 0.
- Orchestrator status was `failed` with `evidence_not_found`: its evidence check resolves file paths
  relative to the agent's own worktree (`<AGENT_LAB_ROOT>`), so files of an external repository
  are never "found". The CLI call itself succeeded and returned findings. This is a limitation of the
  orchestrator for cross-repository reviews, not of the reviewed code.
- No file was modified by the reviewer (read-only mode; isolation check: no violations).

Each finding was checked by the coordinator (Claude) before acting on it:

| # | Finding (Kiro) | Verdict | Evidence / action |
|---|---|---|---|
| 1 | `enable_constant_velocity(Vector3D(v,0,0))` drives the lead car along world +X, invalidating the lead-braking scenario | **Rejected** | CARLA documents the argument as a velocity in the actor's *local* frame. Empirically, in `lead_vehicle_braking` runs the lead's lateral offset stays near 0 m in the ego frame and AEB-off runs end in a rear-end collision (see `scenario_evaluation.md`) |
| 2 | `delay_ms` is `None` for `aeb_off` while actuation used 0 delay | Rejected | With AEB off no brake command is ever produced, so delay has no meaning; `None` is intentional |
| 3 | AEB `_seen` perception-hold never expires for targets absent from the input list | **Accepted (latent)** | Current runner always passes the same single target, so recorded results are unaffected. Fixed in `aeb.py`; regression test `test_aeb_hold_expires_for_targets_absent_from_input` |
| 4 | NPU bench H2D timing may include residual queued work | Rejected | The previous iteration ends with `.cpu()` on all outputs (blocking D2H), so the stream is idle when H2D timing starts |
| 5 | `get_image` raises on an image newer than the requested frame | Rejected (by design) | In synchronous mode the server produces exactly one image per tick; a newer frame means desynchronisation, and fail-fast is preferred over silently pairing the wrong frame |
