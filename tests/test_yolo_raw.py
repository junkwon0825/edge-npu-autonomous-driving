import numpy as np
import pytest
import torch

from npu_ad.config import load_config, resolve
from npu_ad.perception.yolo_raw import Letterbox, letterbox, scale_boxes, to_tensor

WEIGHTS = resolve(load_config()["model"]["weights"])


def test_letterbox_fixed_square_and_inverse_mapping():
    img = np.zeros((450, 800, 3), np.uint8)
    out, lb = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert lb.gain == pytest.approx(0.8)
    assert lb.pad_x == 0 and lb.pad_y == 140
    det = torch.tensor([[0.0, 140.0, 640.0, 500.0, 0.9, 2.0]])
    back = scale_boxes(det, lb, 800, 450)
    assert back[0, :4] == pytest.approx([0, 0, 800, 450])
    assert scale_boxes(torch.zeros(0, 6), Letterbox(1, 0, 0, 640), 800, 450).shape == (0, 6)


def test_to_tensor_rgb_order_and_range():
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0] = 255  # blue in BGR
    x = to_tensor(img)
    assert x.shape == (1, 3, 4, 4) and x.dtype == torch.float32
    assert x[0, 2].max() == 1.0 and x[0, 0].max() == 0.0  # blue ends up in channel 2 (RGB)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="models/yolo26n.pt not present")
def test_raw_split_matches_ultralytics_forward_cpu():
    from npu_ad.perception.yolo_raw import RawDecoder, load_raw_model

    raw, head = load_raw_model(str(WEIGHTS))
    dec = RawDecoder(head, 640)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        ref = raw.model(x)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        b, s = raw(x)
    assert b.shape == (1, 4 * head.reg_max, 8400) and s.shape == (1, head.nc, 8400)
    assert (dec.decode(b, s) - ref).abs().max().item() < 1e-4
