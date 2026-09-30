"""Parity tests: hotcoco.mask vs pycocotools.mask.

Every function in hotcoco.mask that has a pycocotools equivalent must produce
identical output (same types, same values).
"""

import numpy as np
import pycocotools.mask as pm
import pytest
from hotcoco import mask as hm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mask_2d():
    """(10, 10) Fortran-order binary mask with a 3×4 rectangle."""
    m = np.zeros((10, 10), dtype=np.uint8, order="F")
    m[2:5, 3:7] = 1
    return m


@pytest.fixture()
def mask_3d():
    """(10, 10, 2) Fortran-order binary mask with two distinct regions."""
    m = np.zeros((10, 10, 2), dtype=np.uint8, order="F")
    m[2:5, 3:7, 0] = 1
    m[0:2, 0:3, 1] = 1
    return m


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


class TestEncode:
    def test_2d_fortran(self, mask_2d):
        rp = pm.encode(mask_2d)
        rh = hm.encode(mask_2d)
        assert rp["counts"] == rh["counts"]
        assert list(rp["size"]) == list(rh["size"])

    def test_3d_fortran(self, mask_3d):
        rp = pm.encode(mask_3d)
        rh = hm.encode(mask_3d)
        assert len(rp) == len(rh) == 2
        for i, (a, b) in enumerate(zip(rp, rh)):
            assert a["counts"] == b["counts"], f"slice {i}"
            assert list(a["size"]) == list(b["size"]), f"slice {i}"

    def test_2d_c_order(self, mask_2d):
        """C-order arrays should produce the same RLE as Fortran-order."""
        m_c = np.ascontiguousarray(mask_2d)
        rp = pm.encode(np.asfortranarray(m_c))
        rh = hm.encode(m_c)
        assert rp["counts"] == rh["counts"]

    def test_3d_c_order(self, mask_3d):
        """C-order 3D arrays should produce the same RLE as Fortran-order."""
        m_c = np.ascontiguousarray(mask_3d)
        rp = pm.encode(np.asfortranarray(m_c))
        rh = hm.encode(m_c)
        assert len(rp) == len(rh)
        for i, (a, b) in enumerate(zip(rp, rh)):
            assert a["counts"] == b["counts"], f"slice {i}"

    def test_return_type_2d(self, mask_2d):
        rh = hm.encode(mask_2d)
        assert isinstance(rh, dict)
        assert isinstance(rh["counts"], bytes)
        assert isinstance(rh["size"], list)

    def test_return_type_3d(self, mask_3d):
        rh = hm.encode(mask_3d)
        assert isinstance(rh, list)
        assert all(isinstance(r, dict) for r in rh)

    def test_all_zeros(self):
        m = np.zeros((5, 5), dtype=np.uint8, order="F")
        rp = pm.encode(m)
        rh = hm.encode(m)
        assert rp["counts"] == rh["counts"]

    def test_all_ones(self):
        m = np.ones((5, 5), dtype=np.uint8, order="F")
        rp = pm.encode(m)
        rh = hm.encode(m)
        assert rp["counts"] == rh["counts"]


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


class TestDecode:
    def test_single_roundtrip(self, mask_2d):
        rle = pm.encode(mask_2d)
        dp = pm.decode(rle)
        dh = hm.decode(rle)
        np.testing.assert_array_equal(dp, dh)
        assert dh.flags.f_contiguous

    def test_list_roundtrip(self, mask_3d):
        rles = pm.encode(mask_3d)
        dp = pm.decode(rles)
        dh = hm.decode(rles)
        np.testing.assert_array_equal(dp, dh)
        assert dh.flags.f_contiguous

    def test_shape_2d(self, mask_2d):
        rle = pm.encode(mask_2d)
        dh = hm.decode(rle)
        assert dh.shape == (10, 10)

    def test_shape_3d(self, mask_3d):
        rles = pm.encode(mask_3d)
        dh = hm.decode(rles)
        assert dh.shape == (10, 10, 2)


# ---------------------------------------------------------------------------
# area
# ---------------------------------------------------------------------------


class TestArea:
    def test_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        assert pm.area(rle) == hm.area(rle)

    def test_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        np.testing.assert_array_equal(pm.area(rles), hm.area(rles))

    def test_return_type_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        a = hm.area(rle)
        assert isinstance(a, (int, np.integer))

    def test_return_type_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        a = hm.area(rles)
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.uint32


# ---------------------------------------------------------------------------
# toBbox
# ---------------------------------------------------------------------------


class TestToBbox:
    def test_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        np.testing.assert_array_equal(pm.toBbox(rle), hm.toBbox(rle))

    def test_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        np.testing.assert_array_equal(pm.toBbox(rles), hm.toBbox(rles))

    def test_snake_case_alias(self, mask_2d):
        rle = pm.encode(mask_2d)
        np.testing.assert_array_equal(hm.to_bbox(rle), hm.toBbox(rle))

    def test_return_type_single(self, mask_2d):
        rle = pm.encode(mask_2d)
        b = hm.toBbox(rle)
        assert isinstance(b, np.ndarray)
        assert b.shape == (4,)

    def test_return_type_list(self, mask_3d):
        rles = pm.encode(mask_3d)
        b = hm.toBbox(rles)
        assert isinstance(b, np.ndarray)
        assert b.shape == (2, 4)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_union(self, mask_3d):
        rles = pm.encode(mask_3d)
        mp = pm.merge(rles, intersect=False)
        mh = hm.merge(rles, intersect=False)
        assert mp["counts"] == mh["counts"]

    def test_intersection(self, mask_3d):
        rles = pm.encode(mask_3d)
        mp = pm.merge(rles, intersect=True)
        mh = hm.merge(rles, intersect=True)
        assert mp["counts"] == mh["counts"]


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------


class TestIou:
    def test_self_iou(self, mask_3d):
        rles = pm.encode(mask_3d)
        iou_p = pm.iou(rles, rles, [False, False])
        iou_h = hm.iou(rles, rles, [False, False])
        np.testing.assert_allclose(iou_p, iou_h, atol=1e-10)

    def test_return_type(self, mask_3d):
        rles = pm.encode(mask_3d)
        iou_h = hm.iou(rles, rles, [False, False])
        assert isinstance(iou_h, np.ndarray)
        assert iou_h.shape == (2, 2)
        assert iou_h.dtype == np.float64


# ---------------------------------------------------------------------------
# frPyObjects
# ---------------------------------------------------------------------------


class TestFrPyObjects:
    def test_polygon_list(self):
        poly = [[3.0, 2.0, 7.0, 2.0, 7.0, 5.0, 3.0, 5.0]]
        rp = pm.frPyObjects(poly, 10, 10)
        rh = hm.frPyObjects(poly, 10, 10)
        assert isinstance(rp, list) and isinstance(rh, list)
        assert len(rp) == len(rh) == 1
        assert rp[0]["counts"] == rh[0]["counts"]

    def test_single_rle_dict(self):
        rle_dict = {"size": [10, 10], "counts": [2, 3, 95]}
        rp = pm.frPyObjects(rle_dict, 10, 10)
        rh = hm.frPyObjects(rle_dict, 10, 10)
        assert isinstance(rp, dict) and isinstance(rh, dict)
        assert rp["counts"] == rh["counts"]

    def test_multiple_polygons(self):
        polys = [[3.0, 2.0, 7.0, 2.0, 7.0, 5.0, 3.0, 5.0], [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]]
        rp = pm.frPyObjects(polys, 10, 10)
        rh = hm.frPyObjects(polys, 10, 10)
        assert len(rp) == len(rh) == 2
        for i, (a, b) in enumerate(zip(rp, rh)):
            assert a["counts"] == b["counts"], f"polygon {i}"


# ---------------------------------------------------------------------------
# Roundtrip: encode → decode → encode
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_encode_decode_encode(self, mask_2d):
        """encode → decode → encode should produce the same RLE."""
        rle1 = hm.encode(mask_2d)
        decoded = hm.decode(rle1)
        rle2 = hm.encode(decoded)
        assert rle1["counts"] == rle2["counts"]

    def test_large_mask(self):
        """Larger mask for stress testing."""
        rng = np.random.RandomState(42)
        m = (rng.rand(480, 640) > 0.5).astype(np.uint8)
        m_f = np.asfortranarray(m)
        rp = pm.encode(m_f)
        rh = hm.encode(m_f)
        assert rp["counts"] == rh["counts"]
        np.testing.assert_array_equal(pm.decode(rp), hm.decode(rh))
        assert pm.area(rp) == hm.area(rh)
        np.testing.assert_array_equal(pm.toBbox(rp), hm.toBbox(rh))


# ---------------------------------------------------------------------------
# Randomized differential: every operation, bit for bit, over many shapes
# ---------------------------------------------------------------------------
#
# The cases above pin one shape each. This section drives every operation over
# random masks, including the 1-pixel and single-row/column shapes where an
# off-by-one in the run encoding would hide, and compares bit for bit: RLE is
# integer run lengths and masks are `uint8`, so there is nothing to be tolerant
# about. `iou` returns doubles and is compared exactly for the same reason: both
# sides compute intersection and union as integer areas, so the quotients should
# be identical.
#
# This is also where segmentation's residual parity difference on val2017
# (AP ~1e-5, against bbox's ~1e-14) would have to originate, since the metric
# arithmetic above it is exact.

RANDOM_CASES = 200


def norm_rle(r: dict) -> tuple:
    """RLE dicts as comparable tuples; `counts` may be bytes or str."""
    counts = r["counts"]
    if isinstance(counts, str):
        counts = counts.encode()
    return (tuple(r["size"]), counts)


def random_masks(rng: np.random.Generator, h: int, w: int, n: int) -> np.ndarray:
    """Fortran-order uint8 masks, the layout pycocotools requires.

    Mixes blobs with salt-and-pepper: run-length coding is most likely to diverge
    where runs are long (large blobs) or maximally short (alternating pixels), and
    uniform random noise only ever produces the latter.
    """
    out = np.zeros((h, w, n), dtype=np.uint8, order="F")
    for k in range(n):
        style = rng.integers(0, 4)
        if style == 0:  # solid rectangle
            y0, y1 = sorted(rng.integers(0, h + 1, size=2))
            x0, x1 = sorted(rng.integers(0, w + 1, size=2))
            out[y0:y1, x0:x1, k] = 1
        elif style == 1:  # salt and pepper — shortest possible runs
            out[:, :, k] = rng.integers(0, 2, size=(h, w), dtype=np.uint8)
        elif style == 2:  # a few overlapping blobs
            for _ in range(rng.integers(1, 4)):
                y0, y1 = sorted(rng.integers(0, h + 1, size=2))
                x0, x1 = sorted(rng.integers(0, w + 1, size=2))
                out[y0:y1, x0:x1, k] = 1
        else:  # degenerate: all-empty or all-full
            out[:, :, k] = rng.integers(0, 2, dtype=np.uint8)
    return np.asfortranarray(out)


@pytest.mark.parametrize("case", range(RANDOM_CASES))
def test_every_operation_matches_bit_for_bit(case):
    rng = np.random.default_rng(42 + case)
    h = int(rng.choice([1, 1, 2, 3, 7, 16, 32, 64]))
    w = int(rng.choice([1, 2, 3, 5, 16, 32, 64]))
    n = int(rng.integers(1, 6))
    ctx = f"case{case}({h}x{w}x{n})"

    masks = random_masks(rng, h, w, n)

    # encode + decode + round-trip
    py_rles = pm.encode(masks)
    hc_rles = hm.encode(masks)
    assert len(py_rles) == len(hc_rles), ctx
    for i, (p, c) in enumerate(zip(py_rles, hc_rles)):
        assert norm_rle(p) == norm_rle(c), f"encode {ctx}[{i}]"
    py_dec = pm.decode(py_rles)
    hc_dec = hm.decode(hc_rles)
    assert py_dec.shape == hc_dec.shape, f"decode {ctx}"
    assert np.array_equal(py_dec, hc_dec), f"decode {ctx}: {int((py_dec != hc_dec).sum())} pixels differ"
    # Round-trip must be the identity — this is what catches a row/column-major
    # transposition that encode and decode both make, and so cancel out on their
    # own output while disagreeing with pycocotools.
    assert np.array_equal(hc_dec, masks), f"roundtrip {ctx}: {int((hc_dec != masks).sum())} pixels differ"

    # area + toBbox
    assert np.array_equal(np.asarray(pm.area(py_rles)), np.asarray(hm.area(hc_rles))), f"area {ctx}"
    assert np.array_equal(np.asarray(pm.toBbox(py_rles)), np.asarray(hm.toBbox(hc_rles))), f"toBbox {ctx}"

    # iou, under every crowd pattern
    if n >= 2:
        split = max(1, n // 2)
        dt_p, gt_p = py_rles[:split], py_rles[split:]
        dt_h, gt_h = hc_rles[:split], hc_rles[split:]
        for crowd_mode, iscrowd in (
            ("none", [0] * len(gt_p)),
            ("all", [1] * len(gt_p)),
            ("mixed", rng.integers(0, 2, size=len(gt_p)).tolist()),
        ):
            py_i = np.asarray(pm.iou(dt_p, gt_p, iscrowd))
            hc_i = np.asarray(hm.iou(dt_h, gt_h, iscrowd))
            assert py_i.shape == hc_i.shape, f"iou {ctx}/{crowd_mode}"
            assert np.array_equal(py_i, hc_i), f"iou {ctx}/{crowd_mode}: max diff {np.abs(py_i - hc_i).max():.3e}"

    # merge, union and intersection
    for intersect in (False, True):
        p = pm.merge(py_rles, intersect=int(intersect))
        c = hm.merge(hc_rles, intersect=intersect)
        assert norm_rle(p) == norm_rle(c), f"merge {ctx}/intersect={intersect}"

    # The LEB128-ish `counts` string codec, round-tripped through hotcoco.
    for i, r in enumerate(hc_rles):
        s = hm.rle_to_string(r)
        back = hm.rle_from_string(s, r["size"][0], r["size"][1])
        assert norm_rle(back) == norm_rle(r), f"rle_string {ctx}[{i}]"

    # frPyObjects: polygon and bbox rasterization, reachable only through here.
    poly = [float(v) for pair in zip(rng.integers(0, w, size=4), rng.integers(0, h, size=4)) for v in pair]
    p = pm.frPyObjects([poly], h, w)
    c = hm.frPyObjects([poly], h, w)
    assert len(p) == len(c), f"frPyObjects/poly {ctx}"
    for i, (a, b) in enumerate(zip(p, c)):
        assert norm_rle(a) == norm_rle(b), f"frPyObjects/poly {ctx}[{i}] poly={poly}"

    bbox = [float(rng.integers(0, w)), float(rng.integers(0, h)), float(rng.integers(0, w)), float(rng.integers(0, h))]
    # pycocotools routes a length-4 entry to its bbox path, which requires an
    # ndarray; hotcoco accepts a plain list too. Feed each what it takes.
    pb = pm.frPyObjects(np.array([bbox], dtype=np.float64), h, w)
    hb = hm.frPyObjects([bbox], h, w)
    assert len(pb) == len(hb), f"frPyObjects/bbox {ctx}"
    for i, (a, b) in enumerate(zip(pb, hb)):
        assert norm_rle(a) == norm_rle(b), f"frPyObjects/bbox {ctx}[{i}] bbox={bbox}"
