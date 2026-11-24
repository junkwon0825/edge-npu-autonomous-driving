# Lane-segmentation porting to Furiosa RNGD (started in V2)

Goal: experience porting a segmentation network (PyTorch -> graph inspection -> AOT compile ->
unsupported-op analysis -> graph rewrite -> run -> correctness). Task accuracy is out of scope here;
correctness = RNGD output vs the same weights on CPU.

Code: `remote/port_lab.py` (groups `ops`, `fix`, `seg`, `seg2`); run on the RNGD host with
`python scripts/npu_remote.py start -- port_lab --group seg2`. Results: `reports/evidence/v2_npu_port_lab_*.json`.

## V2 status (2026-11-24): not yet running on RNGD
| Step | U-Net (4 levels) | LR-ASPP MobileNetV3-L |
|---|---|---|
| as-is | ConvTranspose / bilinear unsupported | unfused BN not importable |
| BN folded | (folded) | dilated depthwise conv -> Mul verification error |
| exact rewrites (CPU diff <= 1e-7) | `max_pool2d` 16x256x512: no populated tactics | dense dilated kernels: `Inconsistent tensor group offsets` (272 s) |

Exact rewrites (each validated on RNGD in isolation, cosine 1.00000):
bilinear x2 -> nearest x2 + [1,2,1]/4 depthwise smoothing with edge replication;
ConvTranspose(k2,s2) -> 1x1 conv to 4*C + PixelShuffle; dilated conv -> zero-inserted dense kernel.

Next (V3): replace the first full-resolution max-pool by a strided conv or shrink the first stage /
input; then compile, run, compare with CPU, measure latency; then lane geometry from the mask.
See `docs/v2_operator_compatibility.md`.
