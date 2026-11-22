# Portfolio Summary / 포트폴리오 요약

## English

**NPU Auto Driving Project 2: CARLA → YOLO26n → GPU / Furiosa RNGD → perception-gated AEB** (personal project)

- Built a reproducible CARLA 0.9.16 pipeline (synchronous mode, frame-matched sensor queue, tracked actor
  teardown). **TESTED**: 3/3 capture runs of 100 contiguous frames, 0 orphaned actors.
- Diagnosed why earlier RNGD compiles failed (unfused BatchNorm; head concat lowered to `select_scatter`) and
  redesigned the deployment boundary: fused YOLO26n split at the raw detection head, decode/NMS on the host.
  The split reproduces ultralytics' output exactly (max abs diff 0.0 CPU, 3e-8 CUDA). **TESTED**
- Compiled the model AOT for Furiosa RNGD (furiosa-torch 2026.4.0, 1902 s) and ran it on `furiosa:0`:
  **2.11 ms mean compute** per 640×640 frame. Against the RTX 3060 FP32 output on 54 held-out CARLA frames:
  recall 98.8 %, precision 99.2 %, mean IoU 0.991. **TESTED** (agreement with GPU, not ground-truth mAP)
- GPU baseline: 13.5 ms model-only / 17.7 ms end-to-end (eager PyTorch, launch-bound at ~24 % GPU util);
  a free-running CARLA server slows inference by 58 %. **TESTED**
- Closed-loop AEB in 6 CARLA scenarios × 6 latency conditions: without AEB 5/5 hazard scenarios collide;
  with AEB 0/25 collide for delays 0–200 ms. Pedestrian safety margin shrinks from 2.19 m to 0.53 m as delay
  grows to 200 ms. Evaluation exposed a stop-and-go defect in my first controller (v1), fixed in v2 with
  before/after evidence. **TESTED** (simulation; NPU latency injected, not NPU-in-the-loop)
- Independent read-only review by a second coding agent (Kiro) via my multi-agent orchestrator; 1 of 5
  findings accepted, 4 rejected with evidence.

Not done / limitations: INT8 quantization (no vision quantization API in the installed SDK — see
`reports/npu_compilation.md`), ground-truth mAP, NPU-in-the-loop control, power measurement,
compiled-GPU (TensorRT) comparison. No real-vehicle validation.

## 한국어

**NPU Auto Driving Project 2: CARLA → YOLO26n → GPU / Furiosa RNGD → 인지 기반 AEB** (개인 프로젝트)

- CARLA 0.9.16에서 재현 가능한 파이프라인 구축(동기 모드, 프레임 번호 일치 센서 큐, actor 추적·정리).
  **TESTED**: 100프레임 연속 캡처 3/3회 성공, 잔여 actor 0개.
- 기존 RNGD 컴파일 실패 원인(BatchNorm 미융합, head concat의 `select_scatter` lowering 오류)을 로그로
  진단하고, 배포 경계를 재설계: fused YOLO26n을 raw detection head에서 분할, decode/NMS는 host에서 수행.
  분할 결과가 ultralytics 원본과 일치(CPU 최대 오차 0.0, CUDA 3e-8). **TESTED**
- Furiosa RNGD용 AOT 컴파일(furiosa-torch 2026.4.0, 1902초) 후 `furiosa:0`에서 실행:
  640×640 프레임당 **연산 평균 2.11 ms**. RTX 3060 FP32 결과 대비 CARLA 평가 54프레임에서
  recall 98.8 %, precision 99.2 %, 평균 IoU 0.991. **TESTED** (GPU와의 일치도이며 GT 기반 mAP 아님)
- GPU 기준선: model-only 13.5 ms / end-to-end 17.7 ms(eager PyTorch, GPU 사용률 ~24 %로 launch-bound).
  CARLA 서버가 비동기로 돌면 추론이 58 % 느려짐을 측정. **TESTED**
- 6개 시나리오 × 6개 지연 조건 폐루프 AEB: AEB 없이 위험 시나리오 5/5 충돌, AEB 적용 시 지연 0–200 ms에서
  0/25 충돌. 보행자 시나리오 안전 여유가 지연 증가에 따라 2.19 m → 0.53 m로 감소. 평가 과정에서 1차 제어기(v1)의
  stop-and-go 결함을 발견해 v2로 수정, 전후 증거 보존. **TESTED** (시뮬레이션; NPU 지연은 주입 방식이며
  NPU-in-the-loop 아님)
- 자체 멀티 에이전트 오케스트레이터로 두 번째 코딩 에이전트(Kiro)의 읽기 전용 독립 리뷰 수행,
  5건 중 1건 채택·4건은 근거와 함께 기각.

미완료/한계: INT8 양자화(설치된 SDK에 vision 양자화 API 없음 — `reports/npu_compilation.md`), GT 기반 mAP,
NPU-in-the-loop 제어, 전력 측정, TensorRT 등 컴파일된 GPU 경로와의 비교. 실차 검증 없음.
