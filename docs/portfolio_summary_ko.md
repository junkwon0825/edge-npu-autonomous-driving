# 포트폴리오 요약 (한국어)

**CARLA → YOLO26n → RTX 3060 / Furiosa RNGD NPU → 인지 기반 AEB** — 개인 프로젝트, 시뮬레이션 한정.
모든 수치의 근거: `reports/evidence/`.

## 검증된 구현
- 재현 가능한 CARLA 0.9.16 파이프라인: 동기 모드, 프레임 번호 일치 센서 큐, actor 추적·정리(시작·삭제
  실패 시에도 서버 설정 복원), 프로세스 단위 CARLA 서버 독점 잠금, 원자적·재개 가능한 결과 파일,
  JSON 구조화 로그. 단위 테스트 31개 통과(+ live CARLA 테스트 1개는 opt-in), ruff 통과.
- NPU 배포: 기존 RNGD 컴파일 실패 원인(BatchNorm 미융합, head concat lowering) 진단 후 배포 경계 재설계
  (fused 모델을 raw head에서 분할, decode/NMS는 host). 융합 전 원본 모델과 72/72 프레임(348/348 검출)
  일치 검증. `furiosa-torch` 2026.4.0으로 FP32·BF16·W8 변형을 AOT 컴파일하고 `furiosa:0`에서 실행.
- AEB 입력 모드 3종을 엄격히 분리: GT oracle, GT-assisted(검출로 게이트된 GT 거리),
  perception-driven(평지 가정 단안 거리 + 최근접 경로 객체 추적, GT는 평가에만 사용).

## 측정 결과
- 모델 단독, batch 1: GPU eager 13.5 ms, **GPU CUDA Graph 2.60 ms**, **RNGD FP32 2.11 ms**,
  BF16 1.85–2.0 ms. RNGD 지터 표준편차 0.007 ms. 보드 기준 프레임당 에너지 ≈ 113 mJ(RNGD) vs
  ≈ 351 mJ(GPU CUDA Graph).
- NPU–GPU 검출 일치도: recall 0.988 / precision 0.992(FP32). 동일 738프레임 open-loop AEB 재생에서
  7/7 실행 모두 첫 제동 결정 시점 동일.
- 30 km/h 폐루프 AEB, 지연 0–200 ms: AEB 없으면 위험 시나리오 전부 충돌, GT-assisted 0/30,
  perception-driven 0/30 충돌, 무위험 45회 실행에서 오제동 0회. 인지 거리 MAE 0.11–0.79 m,
  접근 속도 MAE 1.2–3.2 m/s(가장 약한 부분).

## 부정적 결과도 그대로 보고
- INT8 연산은 툴체인에서 불가(가정이 아니라 컴파일러 probe로 확인). INT8 weight-only는 가중치 메모리
  49 % 절감하지만 빠르지 않고 정확도 하락(recall 0.957).
- 데스크톱에서 원격 NPU로 원본 프레임 전송 121 ms → AEB 실시간 불가.
- 속도 스윕: 40 km/h 보행자 횡단은 모든 모드에서 충돌, 50 km/h에서는 AEB가 오히려 측면 충돌을 유발
  ("곧 지나갈 대상" 판단 로직 부재) — 튜닝으로 숨기지 않고 문서화.
- 실험 무결성: 매트릭스 실행 중 live 테스트가 공유 시뮬레이터에 접속 → 겹친 6개 실행을 격리·재실행
  (실제로 1개가 달라짐), 이후 서버 잠금으로 재발 방지.

## 한계 / 향후 과제
시뮬레이션 한정, 맵·시드 1개, GT 라벨 없음(mAP 아닌 일치도), NPU는 폐루프 밖, NPU batch > 1 미컴파일.
다음 단계: mAP용 라벨셋, 동일 장비 내 NPU, AEB 경로 예측, Kalman 기반 접근 속도, TensorRT 비교.
