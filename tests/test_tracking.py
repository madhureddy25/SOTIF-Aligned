import numpy as np

from sotif_miner.perception.tracking import CMCSort, GlobalMotion, iou_matrix
from sotif_miner.schemas import Detection


def _det(x, y, w=40, h=80, s=0.9, cls="pedestrian"):
    return Detection(box=(x, y, x + w, y + h), score=s, cls=cls)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], float)
    m = iou_matrix(a, b)
    assert m[0, 0] == 1.0 and abs(m[0, 1] - 25 / 175) < 1e-9 and m[0, 2] == 0.0


def test_id_kept_through_short_occlusion():
    tr = CMCSort(motion_compensation=False, lost_buffer=15)
    ids = []
    for f in range(40):
        dets = [] if 15 <= f < 22 else [_det(100 + 3 * f, 100)]
        out, _ = tr.update(dets, None, f)
        ids += [o.track_id for o in out]
    assert len(set(ids)) == 1
    # the reacquisition after the 7-frame gap is reported
    tr2 = CMCSort(motion_compensation=False, lost_buffer=15)
    gaps = []
    for f in range(40):
        out, _ = tr2.update([] if 15 <= f < 22 else [_det(100 + 3 * f, 100)], None, f)
        gaps += [o.reacquired_gap for o in out if o.reacquired_gap]
    assert gaps == [7]


def test_low_confidence_second_pass_keeps_track():
    tr = CMCSort(motion_compensation=False)
    ids = set()
    for f in range(20):
        s = 0.35 if 8 <= f < 12 else 0.9  # flickers into the low band
        out, _ = tr.update([_det(100 + 2 * f, 100, s=s)], None, f)
        ids |= {o.track_id for o in out}
    assert ids == {1}


def test_termination_reported():
    tr = CMCSort(motion_compensation=False, lost_buffer=3)
    term = []
    for f in range(15):
        out, t = tr.update([_det(200, 100)] if f < 6 else [], None, f)
        term += t
    assert term == [1]


def test_global_motion_recovers_pan():
    rng = np.random.default_rng(0)
    base = (rng.random((240, 400)) * 255).astype(np.uint8)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 1.5)
    img0 = cv2.cvtColor(base[:, :320], cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(base[:, 8:328], cv2.COLOR_GRAY2BGR)  # camera pans right -> scene moves 8 px left
    gm = GlobalMotion(downscale=1.0)
    gm.estimate(img0)
    A = gm.estimate(img1)
    assert abs(A[0, 2] + 8) < 1.0 and abs(A[1, 2]) < 1.0
