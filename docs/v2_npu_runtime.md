# V2: NPU runtime (unified backends, remote server, failover, health)

Code: `src/npu_ad/runtime/` and `remote/npu_server.py`. Tests: `tests/test_v2_runtime.py` (13 tests,
including a protocol-compatible fake server, `tests/fake_npu_server.py`).

## Unified inference interface
```python
class InferenceBackend(Protocol):
    name: str; precision: str
    def infer(self, frame_id, capture_t, bgr) -> InferenceResult: ...
    def health(self) -> dict: ...
    def close(self) -> None: ...
```
`InferenceResult` carries:
- frame id and capture timestamp
- backend name, precision, model version (`yolo26n sha256:9b09cc8b`) and input shape
- `ok` / `error`
- detections in image pixels
- wall latency, start and end timestamps on the pipeline clock
- per-stage durations

Tracker, filters and AEB only consume `InferenceResult`, never a device API.

| Backend | Role |
|---|---|
| `GpuInferenceBackend` | V1 CUDA path (`GpuDetector`): baseline, fallback, replay validator |
| `FuriosaRemoteBackend` | client of the persistent RNGD server (BF16 by default) |
| `ReplayInferenceBackend` | recorded detections looked up by frame id (open-loop replay) |
| `MockBackend` | deterministic latency and failure injection for tests |
| `FailoverBackend` | wraps primary (RNGD) and secondary (GPU), see below |

## Remote server (`remote/npu_server.py`)
- **Session.** One persistent process per session, started through the existing SSH alias; nothing
  is installed system-wide and no port is opened.
- **Model.** It loads the BF16 module from the SDK compile cache (5 s, no recompilation), checks the
  input shape and dtype, and warms up.
- **Protocol.** Requests and responses are binary and length-prefixed:
  - `INFER` (raw letterbox, content-only raw, or JPEG/PNG)
  - `ECHO` (transport only)
  - `HEALTH` (`furiosa-smi` power and temperature, counters, load average)
  - `STALL` (fault injection: the server sleeps)
  - `QUIT`
- **Per request.** Pre-processing, H2D, synchronised execution, D2H, exact pre-filtered decode and NMS,
  and a per-stage timing record.
- **Errors.** Exceptions are returned as status 1 with the message; the server keeps serving.
- **Tuning.** `--threads` pins the torch CPU threads; see `docs/v2_npu_transport.md` for why this
  matters on a shared host.

## Client (`FuriosaRemoteBackend`)
- **Reading.** A reader thread demultiplexes responses by frame id. A late answer to a timed-out
  request is counted as `n_stale` and discarded, never returned as current.
- **Writing.** A writer thread sends requests:
  - `infer()` has one deadline covering send and receive (`timeout_s`).
  - If the previous send is still blocked (stalled server, congested link), the request fails at once
    with `TransportBusy` (backpressure) instead of hanging.
  - The original blocking send could hang for the full duration of a server stall, which a unit test
    exposed.
- **Failures.** They come back as `ok=False` results with the error. Counters: `n_ok`, `n_err`,
  `n_timeout`, `n_stale`, `n_busy`.

## Failover (`FailoverBackend`) and health
States:
- `PRIMARY` (RNGD healthy)
- `FALLBACK`: after `fail_threshold` consecutive failures or timeouts, the failing frame is re-run on
  the GPU immediately. Every `probe_interval_s`, one frame is tried on the RNGD again; success returns
  to PRIMARY.
- `DEGRADED`: both backends fail. The result is `ok=False` and the controller holds its last command
  (`docs/v2_aeb_controller.md`).

Every transition is logged as a structured event with time, frame id, from/to and reason. Every result
names the backend that produced it, so a device switch is never silent.

`HealthMonitor` records client counters on every call. It queries the remote device at most once per
`remote_every_s` (default 3–10 s), so the shared host is not polled aggressively.

Measured failover behaviour in the live loop: `docs/v2_benchmark.md` § live loop.

## Multi-PE
`furiosa:0..7` are single-PE devices of one RNGD. Scaling from 1 to 4 PEs reaches 3.3× throughput;
8 threads become host-bound. One `CompileModule` instance is needed per device. See
`docs/v2_operator_compatibility.md` § 5.
