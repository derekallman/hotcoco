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


# `mask.area` sums the foreground runs as the compressed `counts` string
# decodes, without building a run list. These masks aim at the decoder's
# corners: runs long enough to need five or more 5-bit groups (past 2^20),
# including a negative multi-group delta; masks of one-pixel runs, where nearly
# every value goes through the stride-2 delta; and the empty, full, and
# single-pixel extremes.


def _mask_from_runs(runs: list[int], h: int, w: int) -> np.ndarray:
    """Fortran-order mask whose column-major run lengths are `runs`, background first."""
    assert sum(runs) == h * w
    flat = np.repeat(np.arange(len(runs)) % 2, runs).astype(np.uint8)
    return np.asfortranarray(flat.reshape((h, w), order="F"))


def _runs_of(m: np.ndarray) -> list[int]:
    """The uncompressed RLE counts of a mask: column-major, background first."""
    flat = m.ravel(order="F")
    edges = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], edges, [flat.size]])).tolist()
    return [0, *runs] if flat[0] else runs


def _area_masks() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    h, w = 427, 640
    first, last = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    first[0, 0] = last[-1, -1] = 1
    middle = np.zeros((h, w), np.uint8)
    middle[200, 300] = 1
    masks = {
        "empty": np.zeros((h, w), np.uint8),
        "full": np.ones((h, w), np.uint8),
        "pixel_first": first,
        "pixel_middle": middle,
        "pixel_last": last,
        "one_by_one": np.ones((1, 1), np.uint8),
        # Column-major, an odd height makes every run one pixel long.
        "checkerboard": (np.indices((h, w)).sum(axis=0) % 2).astype(np.uint8),
        "long_runs": _mask_from_runs([1_200_000, 800_000, 250_000], 1500, 1500),
        # Run 3 is a delta of 3 - 1_050_000 against run 1.
        "long_negative_delta": _mask_from_runs([1_100_000, 1_050_000, 2, 3, 95_000, 4_995], 1500, 1500),
    }
    for density in (0.001, 0.05, 0.5, 0.95, 0.999):
        masks[f"random_{density}"] = (rng.random((h, w)) < density).astype(np.uint8)
    return {name: np.asfortranarray(m) for name, m in masks.items()}


AREA_MASKS = _area_masks()


@pytest.mark.parametrize("name", list(AREA_MASKS))
def test_area_from_compressed_counts_matches_pycocotools(name):
    m = AREA_MASKS[name]
    rle = pm.encode(m)
    want = int(pm.area(rle))
    as_str = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
    for spelling in (rle, as_str):
        got = hm.area(spelling)
        assert type(got) is int
        assert got == want


@pytest.mark.parametrize("name", list(AREA_MASKS))
def test_area_from_uncompressed_counts_matches_pycocotools(name):
    # pycocotools' `area` takes only compressed counts; `frPyObjects`
    # compresses the run list for it, and proves the list is the same mask.
    m = AREA_MASKS[name]
    rle = pm.encode(m)
    h, w = m.shape
    uncompressed = {"size": [h, w], "counts": _runs_of(m)}
    assert pm.frPyObjects(uncompressed, h, w)["counts"] == rle["counts"]
    assert hm.area(uncompressed) == int(pm.area(rle))


def test_area_batched_matches_pycocotools():
    rles = [pm.encode(m) for m in AREA_MASKS.values()]
    rles += [{"size": r["size"], "counts": r["counts"].decode("ascii")} for r in rles]
    got, want = hm.area(rles), pm.area(rles)
    assert got.dtype == want.dtype == np.uint32
    np.testing.assert_array_equal(got, want)


class TestCountsSpellings:
    """Every RLE dict reader takes the same `counts` spellings. Read as a list
    of ints, the bytes of a compressed string would be run lengths — a wrong
    mask with no error, the issue #5 failure class."""

    @staticmethod
    def _mask_and_rle():
        m = np.zeros((10, 10), np.uint8, order="F")
        m[2:5, 3:7] = 1
        return m, pm.encode(m)

    @pytest.mark.parametrize("wrap", [bytearray, memoryview])
    def test_byte_buffers_are_rejected(self, wrap):
        # pycocotools raises TypeError for these too.
        _, rle = self._mask_and_rle()
        bad = {"size": rle["size"], "counts": wrap(rle["counts"])}
        for fn in (hm.area, hm.decode, hm.toBbox):
            with pytest.raises(TypeError, match="must be str, bytes, or a list of ints"):
                fn(bad)

    @pytest.mark.parametrize("spelling", ["bytes", "str"])
    def test_h_w_spelling_decodes_compressed_counts(self, spelling):
        m, rle = self._mask_and_rle()
        counts = rle["counts"] if spelling == "bytes" else rle["counts"].decode("ascii")
        hw = {"h": 10, "w": 10, "counts": counts}
        assert hm.area(hw) == int(m.sum())
        np.testing.assert_array_equal(hm.decode(hw), m)


@pytest.mark.parametrize(
    ("size", "counts"),
    [
        ([10, 10], b"!"),  # below '0'
        ([10, 10], b"O"),  # negative run
        ([10, 10], "O"),
        ([2, 2], b"5"),  # runs past h * w
        ([10, 10], b"PPPPPP8"),  # run past u32::MAX
        ([10, 10], b"P" * 20),  # too many continuation characters
        ([10, 10], b"\xff"),  # not UTF-8
    ],
)
def test_area_rejects_what_decode_rejects(size, counts):
    rle = {"size": size, "counts": counts}
    with pytest.raises(ValueError) as decode_error:
        hm.decode(rle)
    message = str(decode_error.value)
    for call in (lambda: hm.area(rle), lambda: hm.area([rle])):
        with pytest.raises(ValueError) as area_error:
            call()
        assert str(area_error.value) == message


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


# ---------------------------------------------------------------------------
# encode: the word-at-a-time scan, every memory layout, every one-byte dtype
# ---------------------------------------------------------------------------
#
# `encode` finds where each run ends 32 and 8 bytes at a time, reads a
# Fortran-order mask straight from its buffer, and moves a C-order mask to
# Fortran order with an 8x8-block transpose. These cases aim at each of those:
# runs that end on and across word boundaries, sizes that leave a ragged tail
# or a ragged block edge, and every layout path for 2-D and 3-D input.
#
# The reference is pycocotools on the *binarized* mask: hotcoco treats any
# nonzero byte as foreground, while pycocotools starts a new run wherever the
# byte value changes (a 1 next to a 2).

SCAN_SHAPES = [(1, 1), (1, 7), (7, 1), (3, 5), (8, 8), (9, 7), (5, 13), (16, 2), (33, 3), (11, 37), (64, 65)]


def reference_rle(m: np.ndarray):
    """pycocotools' RLE of ``m != 0``; a list of RLEs for an (H, W, N) stack."""
    return pm.encode(np.asfortranarray(m != 0, dtype=np.uint8))


def scan_patterns(h: int, w: int, rng: np.random.Generator):
    """Named (H, W) uint8 0/1 masks, built in column-major order — the order
    the scan reads — so the run boundaries land where they are aimed."""
    n = h * w
    index = np.arange(n)
    flat = {
        "zeros": np.zeros(n),
        "ones": np.ones(n),
        "first_pixel": index == 0,
        "last_pixel": index == n - 1,
        "all_but_first": index != 0,
        "all_but_last": index != n - 1,
        "alternating": index % 2,
        "alternating_inverted": 1 - index % 2,
        # Runs ending inside a word, on a word edge, across a word edge, and
        # across the 32-byte block edge; the slices clip on small masks.
        "boundary_runs": np.isin(index, np.r_[5:11, 14:16, 16:18, 30:34, 39:41, 62:67]),
    }
    for p in (0.05, 0.5, 0.95):
        flat[f"random_{p}"] = rng.random(n) < p
    for name, f in flat.items():
        yield name, np.asarray(f, dtype=np.uint8).reshape((h, w), order="F")


def layouts_2d(m: np.ndarray):
    """The same (H, W) mask under every memory layout encode has a path for."""
    h, w = m.shape
    yield "fortran", np.asfortranarray(m)
    yield "c_order", np.ascontiguousarray(m)
    big = np.zeros((2 * h + 1, 3 * w + 2), dtype=m.dtype)
    big[1::2, 2::3] = m
    yield "strided", big[1::2, 2::3]
    yield "reversed", np.ascontiguousarray(m[::-1, ::-1])[::-1, ::-1]
    tall = np.zeros((h + 3, w), dtype=m.dtype, order="F")
    tall[1 : h + 1] = m
    yield "fortran_rows_sliced", tall[1 : h + 1]


def read_only(arr: np.ndarray) -> np.ndarray:
    """A read-only copy of `arr`, in its layout. A copy keeps a `bool` array's
    raw bytes, 2 and 255 included."""
    arr = arr.copy(order="K")
    arr.setflags(write=False)
    return arr


# The layouts the one-byte dtypes are checked in. encode takes no mutable
# borrow, so a read-only array encodes too.
DTYPE_LAYOUTS = {
    "fortran": np.asfortranarray,
    "c_order": np.ascontiguousarray,
    "read_only_fortran": lambda m: read_only(np.asfortranarray(m)),
    "read_only_c_order": lambda m: read_only(np.ascontiguousarray(m)),
}


class TestEncodeScan:
    @pytest.mark.parametrize("shape", SCAN_SHAPES)
    def test_2d_every_layout_matches_pycocotools(self, shape):
        rng = np.random.default_rng(shape[0] * 100 + shape[1])
        for name, m in scan_patterns(*shape, rng):
            ref = norm_rle(reference_rle(m))
            for layout, view in layouts_2d(m):
                assert np.array_equal(view, m)
                assert norm_rle(hm.encode(view)) == ref, f"{shape} {name} {layout}"

    @pytest.mark.parametrize("shape", SCAN_SHAPES)
    def test_2d_every_one_byte_dtype_matches_pycocotools(self, shape):
        rng = np.random.default_rng(shape[0] * 100 + shape[1] + 1)
        for name, m in scan_patterns(*shape, rng):
            fg = m != 0
            mixed = np.where(fg, rng.integers(1, 256, m.shape), 0).astype(np.uint8)
            variants = {
                "bool": fg,
                # A bool array viewed from uint8 holds whatever bytes it had:
                # any nonzero one is foreground.
                "bool_bytes_mixed": mixed.view(bool),
                "uint8_2": fg.astype(np.uint8) * 2,
                "uint8_255": fg.astype(np.uint8) * 255,
                # int8 has no pycocotools path (it rejects the dtype), and -1
                # reaches the scan as 255.
                "int8_minus_one": fg.astype(np.int8) * -1,
                "int8_mixed": np.where(fg, rng.choice(np.array([1, -1, 127, -128], dtype=np.int8), m.shape), 0),
                "uint8_mixed": mixed,
            }
            ref = norm_rle(reference_rle(m))
            for dtype_name, raw in variants.items():
                for layout, make in DTYPE_LAYOUTS.items():
                    assert norm_rle(hm.encode(make(raw))) == ref, f"{shape} {name} {dtype_name} {layout}"

    @pytest.mark.parametrize("n", [1, 3, 8, 9, 17])
    @pytest.mark.parametrize("shape", [(1, 1), (7, 3), (8, 8), (9, 5), (17, 2)])
    def test_3d_every_layout_matches_pycocotools(self, shape, n):
        rng = np.random.default_rng(shape[0] * 1000 + shape[1] * 10 + n)
        patterns = [m for _, m in scan_patterns(*shape, rng)]
        stack = np.stack([patterns[i % len(patterns)] for i in range(n)], axis=2)
        ref = [norm_rle(r) for r in reference_rle(stack)]
        wide = np.zeros(stack.shape[:2] + (2 * n,), dtype=np.uint8)
        wide[:, :, ::2] = stack
        for layout, view in {
            "fortran": np.asfortranarray(stack),
            "c_order": np.ascontiguousarray(stack),
            "strided_n": wide[:, :, ::2],
            "c_order_bool": np.ascontiguousarray(stack != 0),
            "c_order_bool_bytes": np.ascontiguousarray(stack * 7).view(bool),
            "fortran_uint8_255": np.asfortranarray(stack * 255),
            "fortran_int8_minus_one": np.asfortranarray(stack.astype(np.int8) * -1),
            "read_only_fortran_bool": read_only(np.asfortranarray(stack * 2).view(bool)),
            "read_only_c_order_int8": read_only(np.ascontiguousarray(stack.astype(np.int8) * -1)),
        }.items():
            got = hm.encode(view)
            assert isinstance(got, list) and len(got) == n
            assert [norm_rle(r) for r in got] == ref, f"{shape}x{n} {layout}"
            # Each (H, W) slice on its own: a strided view of the stack.
            assert [norm_rle(hm.encode(view[:, :, i])) for i in range(n)] == ref, f"{shape}x{n} {layout} slices"

    def test_full_size_stack_matches_pycocotools(self):
        """Realistic size, with H (426) and N (10) off the 8x8 block grid."""
        rng = np.random.default_rng(7)
        stack = np.zeros((426, 640, 10), dtype=bool)
        for i in range(10):
            for _ in range(rng.integers(1, 4)):
                y0, x0 = rng.integers(0, 400), rng.integers(0, 600)
                stack[y0 : y0 + rng.integers(1, 200), x0 : x0 + rng.integers(1, 300), i] = True
        stack[:, :, 9] = rng.random((426, 640)) < 0.5
        ref = [norm_rle(r) for r in reference_rle(stack)]
        for layout in (np.asfortranarray, np.ascontiguousarray):
            assert [norm_rle(r) for r in hm.encode(layout(stack))] == ref, layout.__name__
        # TorchMetrics: an (N, H, W) bool batch, one Fortran-order mask at a time.
        for i, m in enumerate(np.ascontiguousarray(stack.transpose(2, 0, 1))):
            assert norm_rle(hm.encode(np.asfortranarray(m))) == ref[i]

    @pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0), (0, 5, 2), (5, 0, 2), (3, 4, 0), (0, 3, 2)])
    def test_zero_size_matches_pycocotools(self, shape):
        m = np.zeros(shape, dtype=np.uint8)
        ref = pm.encode(np.asfortranarray(m))
        for layout in (np.asfortranarray, np.ascontiguousarray):
            got = hm.encode(layout(m))
            if len(shape) == 2:
                assert norm_rle(got) == norm_rle(ref)
            else:
                assert [norm_rle(r) for r in got] == [norm_rle(r) for r in ref]


# ---------------------------------------------------------------------------
# frPyObjects: dispatch and far out-of-image polygons (2026-10 review)
# ---------------------------------------------------------------------------


class TestFrPyObjectsDispatch:
    """pycocotools picks box vs polygon once, from ``len(pyobj[0])``."""

    @pytest.mark.parametrize(
        "polys",
        [
            # A 4-value entry after a polygon is a two-point polygon (area 0),
            # not a box: pycocotools gives areas [21, 0].
            [[1, 1, 8, 1, 8, 8], [5, 5, 6, 6]],
            [[1, 1, 8, 1, 8, 8], [1, 1]],
            [[1, 1, 8, 1, 8, 8], [1, 1, 8, 1, 8, 8, 1, 8]],
            # Odd length: pycocotools takes len // 2 points.
            [[1, 1, 8, 1, 8, 8], [1, 1, 8, 1, 8, 8, 1]],
        ],
    )
    def test_first_entry_decides_for_the_list(self, polys):
        rp = pm.frPyObjects(polys, 10, 10)
        rh = hm.frPyObjects(polys, 10, 10)
        assert [int(a) for a in pm.area(rp)] == [int(a) for a in hm.area(rh)]
        assert [r["counts"] for r in rp] == [r["counts"] for r in rh]

    def test_box_first_makes_every_entry_a_box(self):
        # pycocotools' box path needs an ndarray (rows must then all be 4 wide);
        # hotcoco also takes a list, and rejects a non-box entry in it.
        boxes = [[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]]
        rp = pm.frPyObjects(np.array(boxes), 10, 10)
        rh = hm.frPyObjects(boxes, 10, 10)
        assert [r["counts"] for r in rp] == [r["counts"] for r in rh]
        with pytest.raises(ValueError, match="every entry must be"):
            hm.frPyObjects([[1.0, 1.0, 3.0, 3.0], [1, 1, 8, 1, 8, 8]], 10, 10)

    def test_short_first_entry_is_rejected(self):
        with pytest.raises(Exception):  # noqa: B017 - pycocotools raises bare Exception
            pm.frPyObjects([[1, 1], [1, 1, 8, 1, 8, 8]], 10, 10)
        with pytest.raises(ValueError):
            hm.frPyObjects([[1, 1], [1, 1, 8, 1, 8, 8]], 10, 10)


@pytest.mark.parametrize(
    ("poly", "h", "w"),
    [
        ([0, 0, 300, 100, 0, 100], 100, 100),
        ([-500, -20, 50, 40, 600, 90, 20, 130], 100, 100),
        ([-250, 50, 50, -400, 350, 50, 50, 450], 100, 100),
        ([10, 10, 5000, 20, 30, 5000], 64, 48),
        ([-3000, -3000, 3000, -2900, -2950, 3000], 50, 80),
        ([1, 1, 1e6, 2, 3, 1e6], 100, 100),
        ([10, 10, 1e6, 200, 300, 400], 480, 640),
        ([10, 10, 200, 1e6, 300, 400], 480, 640),
        ([-1e6, -1e6, 1e6, 5, 300, 1e6], 480, 640),
        ([0, -200, 60, 90, -100, 90], 80, 60),
        ([5, 5, 95, -295, 95, 95], 100, 100),
    ],
)
def test_far_out_of_image_polygon_matches_pycocotools(poly, h, w):
    """Vertices far outside the image are not clamped, so edge slopes hold."""
    rp = pm.frPyObjects([poly], h, w)[0]
    rh = hm.frPyObjects([poly], h, w)[0]
    assert int(pm.area(rp)) == int(hm.area(rh))
    assert rp["counts"] == rh["counts"]
