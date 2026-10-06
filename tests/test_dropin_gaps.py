"""Regression tests for the 1.0 drop-in compatibility gaps (PyO3 layer).

Each test here pins a fix from the 1.0 preflight review: derived datasets
carrying the full API, custom-key round-tripping, pycocotools-format annToRLE,
pycocotools stats semantics, real Python warnings, and error types. None of
them need the gitignored data/ directory.
"""

import enum
import json
import re
import sys
import warnings

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval, LVISeval, LVISResults, mask


def tiny_dataset():
    return {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }


def tiny_dt(gt):
    return gt.load_res([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9}])


# ---------------------------------------------------------------------------
# Derived datasets carry the full API (subclass-loss fix)
# ---------------------------------------------------------------------------


class TestDerivedDatasetsKeepFullApi:
    def test_split_results_have_browse(self):
        coco = COCO(tiny_dataset())
        train, val = coco.split(val_frac=0.5)
        assert hasattr(type(train), "browse")
        assert callable(type(val).browse)

    def test_all_derivation_paths_have_browse(self):
        coco = COCO(tiny_dataset())
        derived = [coco.filter(cat_ids=[1]), coco.sample(n=1), tiny_dt(coco), COCO.merge([coco])]
        for d in derived:
            assert hasattr(type(d), "browse"), f"{type(d)} lost browse()"

    def test_browse_without_deps_or_dir_raises_cleanly(self):
        """browse() reaches the Python implementation (not AttributeError)."""
        coco = COCO(tiny_dataset())
        with pytest.raises((ValueError, ImportError)):
            coco.browse()


# ---------------------------------------------------------------------------
# Custom keys survive the Rust round-trip
# ---------------------------------------------------------------------------


class TestCustomKeysRoundTrip:
    def test_annotation_image_category_extras_survive(self):
        ds = tiny_dataset()
        ds["annotations"][0]["confidence_source"] = "human"
        ds["annotations"][0]["tags"] = ["hard", {"nested": 1}]
        ds["images"][0]["weather"] = "rainy"
        ds["categories"][0]["taxonomy_id"] = 42

        coco = COCO(ds)
        out = coco.dataset
        ann = next(a for a in out["annotations"] if a["id"] == 1)
        assert ann["confidence_source"] == "human"
        assert ann["tags"] == ["hard", {"nested": 1}]
        assert next(i for i in out["images"] if i["id"] == 1)["weather"] == "rainy"
        assert next(c for c in out["categories"] if c["id"] == 1)["taxonomy_id"] == 42

    def test_extras_survive_filter(self):
        ds = tiny_dataset()
        ds["annotations"][0]["custom"] = {"a": 1}
        coco = COCO(ds).filter(cat_ids=[1])
        anns = coco.dataset["annotations"]
        assert anns and anns[0]["custom"] == {"a": 1}

    def test_extras_visible_in_load_anns_and_anns(self):
        ds = tiny_dataset()
        ds["annotations"][1]["reviewer"] = "alice"
        coco = COCO(ds)
        assert coco.load_anns(2)[0]["reviewer"] == "alice"
        assert coco.anns[2]["reviewer"] == "alice"

    def test_non_serializable_extra_raises(self):
        ds = tiny_dataset()
        ds["annotations"][0]["bad"] = object()
        with pytest.raises(TypeError):
            COCO(ds)

    def test_slice_by_callable_sees_custom_image_keys(self):
        ds = tiny_dataset()
        ds["images"][0]["weather"] = "rainy"
        ds["images"][1]["weather"] = "sunny"
        gt = COCO(ds)
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.evaluate()
        result = ev.slice_by(lambda img: img.get("weather"))
        assert "rainy" in result and "sunny" in result
        assert result["rainy"]["num_images"] == 1


# ---------------------------------------------------------------------------
# Plain custom values skip json.dumps and land exactly where it would
#
# convert.rs converts a record whose custom values are all None, bool, int,
# float, or str (subclasses included), or exact list, tuple, or str-keyed dict
# of those, in Rust; one value of any other kind sends the whole record
# through json.dumps. The oracle is that path itself: every check compares a
# record against the same record with one extra key whose value the direct
# conversion declines, which forces it through json.dumps. Python's own
# json.loads(json.dumps(v)) is a second, independent oracle.
# ---------------------------------------------------------------------------

FORCE_JSON_PATH = "zz_force_json_path"


class _ListSub(list):
    """json reads a list subclass through `__iter__`."""

    def __iter__(self):
        return iter(["from __iter__"])


class _DictSub(dict):
    """json reads a dict subclass through `items()`."""

    def items(self):
        return [("from_items", 1)]


class _FloatSub(float):
    """json writes `float.__repr__` of the value; none of these count."""

    def __repr__(self):
        return "not a float literal"

    def __str__(self):
        return "not a float literal"

    def __float__(self):
        return 99.0


class _IntSub(int):
    pass


class _IntOverrides(int):
    """json writes `int.__repr__` of the value; none of these count."""

    def __repr__(self):
        return "not an int literal"

    def __str__(self):
        return "not an int literal"

    def __int__(self):
        return 99

    def __index__(self):
        return 99

    def __float__(self):
        return 99.0


class _StrSub(str):
    """json writes the string itself; neither of these counts."""

    def __repr__(self):
        return "'not this string'"

    def __str__(self):
        return "not this string"


class _HashOverride(str):
    """A key that a lookup by its plain text misses: its hash is not `str`'s."""

    def __hash__(self):
        return 12345


class _EqOverride(str):
    """A key that a lookup by its plain text misses: it equals only itself."""

    def __eq__(self, other):
        return self is other

    __hash__ = str.__hash__


class _Level(enum.IntEnum):
    HIGH = 3


# json.dumps writes a list subclass by its own rules; the direct conversion
# takes exact containers only, so a record carrying this goes through json.dumps.
_FORCING_VALUE = _ListSub()
_FORCE = {FORCE_JSON_PATH: _FORCING_VALUE}


@pytest.fixture
def dumps_calls(monkeypatch):
    """The values convert.rs hands to json.dumps, one per call."""
    calls = []
    real = json.dumps

    def counting(obj, *args, **kwargs):
        calls.append(obj)
        return real(obj, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", counting)
    return calls


# What the Rust conversion accepts, at the edges where serde_json's reading
# of json.dumps output could differ from a direct conversion.
FAST_SCALARS = {
    "none": None,
    "true": True,
    "false": False,
    "int_zero": 0,
    "int_one": 1,
    "int_neg": -1,
    "i64_min": -(2**63),
    "i64_max": 2**63 - 1,
    "u64_above_i64": 2**63,
    "u64_max": 2**64 - 1,
    "float_one": 1.0,  # json writes 1.0, which must not come back as the int 1
    "float_zero": 0.0,
    "float_neg_zero": -0.0,
    "float_integral": 2160.0,
    "float_2_53": 2.0**53,
    "float_1e16": 1e16,  # repr is '1e+16': an exponent and no '.'
    "float_1e22": 1e22,
    "float_1e300": 1e300,
    "float_max": sys.float_info.max,
    "float_min_normal": sys.float_info.min,
    "float_subnormal": 5e-324,
    "float_third": 1 / 3,
    "float_tenth": 0.1,
    "float_neg": -12345.678,
    "str_empty": "",
    "str_ascii": "human",
    "str_unicode": "café ☕ 😀 中文",
    "str_escapes": 'quote " backslash \\ newline \n tab \t nul \x00 unit-sep \x1f del \x7f',
    # Subclasses: json writes the base value, whatever the subclass overrides.
    "np_float64": np.float64(1.5),
    "np_str": np.str_("numpy str"),
    "float_sub": _FloatSub(2.5),
    "int_sub": _IntSub(7),
    "int_sub_u64_max": _IntSub(2**64 - 1),
    "int_overrides": _IntOverrides(-7),
    "int_enum": _Level.HIGH,
    "str_sub": _StrSub("sub"),
}


def _nested(levels, leaf=None):
    """`leaf` inside `levels` nested lists."""
    for _ in range(levels):
        leaf = [leaf]
    return leaf


def _circular_list():
    value = [1]
    value.append(value)
    return value


def _circular_dict():
    value = {"a": 1}
    value["self"] = value
    return value


# Containers the direct conversion takes: a tuple is an array, as json.dumps
# writes it, and a dict's keys come back in the order the json.dumps path
# gives them, whatever order they were inserted in. 32 levels is the deepest
# it follows (MAX_JSON_DEPTH in convert.rs).
NESTED_VALUES = {
    "empty_list": [],
    "empty_tuple": (),
    "empty_dict": {},
    "list": [1, 2.0, "x", None, True],
    "dict": {"b": 1, "a": [0.5, {"c": None}]},
    "list_of_scalars": list(FAST_SCALARS.values()),
    "tuple_of_scalars": tuple(FAST_SCALARS.values()),
    "dict_of_scalars": dict(reversed(FAST_SCALARS.items())),
    "unsorted_keys": {"zeta": 1, "alpha": 2.0, "Mid": None, "_": "x", "é": True, "": [], "10": 0, "9": -0.0},
    "cvat_attributes": {"occluded": False, "truncated": True, "tags": ["a", "b"]},
    "mixed": {"z": [(), {}, [[]]], "a": ({"y": (1, 2**64 - 1)}, [-(2**63), 5e-324]), "m": {"k": {"j": "deep"}}},
    "depth_32": _nested(30, {"leaf": (1.0, -0.0)}),  # 30 lists, a dict, and a tuple
    "depth_32_lists": _nested(32, 1e16),
    # Subclasses of float, int, and str, one or more containers down.
    "np_float64_in_dict": {"a": np.float64(1.5)},
    "float_sub_in_tuple": (_FloatSub(2.5),),
    "int_subs_in_list": [1, _IntOverrides(7), _Level.HIGH],
    "str_subs_in_dict": {"a": _StrSub("sub"), "b": np.str_("np")},
    "str_sub_keys": {_StrSub("k"): 1, np.str_("j"): 2},  # json writes the base string
}

# Values the direct conversion does not take. Whichever path a value goes
# down, what comes back must equal what json.dumps gives.
JSON_PATH_VALUES = {
    "int_over_u64": 2**64,  # serde_json reads it back as a float
    "int_under_i64": -(2**63) - 1,
    "int_sub_over_u64": _IntSub(2**64),
    "surrogate_pair": chr(0xD83D) + chr(0xDE00),  # json escapes both halves; serde_json joins them
    # json reads a container subclass through methods the subclass can override.
    "list_sub": _ListSub([1, 2]),
    "dict_sub": _DictSub({"real": 1}),
    # The same, one or more containers down.
    "list_sub_in_dict": {"a": _ListSub([1, 2])},
    "dict_sub_in_list": [_DictSub({"real": 1})],
    "int_over_u64_in_list": [2**64],
    "int_under_i64_in_dict": {"a": -(2**63) - 1},
    "surrogate_pair_in_list": [chr(0xD83D) + chr(0xDE00)],
    # json.dumps spells a non-str key its own way: 1 -> "1", True -> "true".
    "int_key": {1: "a"},
    "float_key": {1.5: "a", 2.0: "b"},
    "bool_key": {True: "a", False: "b"},
    "none_key": {None: "a"},
    "colliding_keys": {"1": "str", 1: "int"},  # two "1" keys in the JSON; the last one wins
    "non_str_key_deep_down": {"a": [{"b": {2: None}}]},
    "depth_33": _nested(33, 1),
    "depth_100": _nested(100, 1),  # still within serde_json's limit of 128
}

# What the json.dumps path rejects, and so must still reject. numpy's float32
# and int64 subclass neither float nor int, so json does not take them.
REJECTED_VALUES = [
    (float("nan"), ValueError, "did not round-trip through JSON"),
    (float("inf"), ValueError, "did not round-trip through JSON"),
    (float("-inf"), ValueError, "did not round-trip through JSON"),
    (np.float64("nan"), ValueError, "did not round-trip through JSON"),
    (chr(0xD800), ValueError, "did not round-trip through JSON"),  # a lone surrogate
    (object(), TypeError, "not JSON serializable"),
    (np.float32(1.5), TypeError, "not JSON serializable"),
    (np.int64(3), TypeError, "not JSON serializable"),
    # The same, one or more containers down.
    ([1.0, float("nan")], ValueError, "did not round-trip through JSON"),
    ({"a": (float("inf"),)}, ValueError, "did not round-trip through JSON"),
    ({"a": [chr(0xD800)]}, ValueError, "did not round-trip through JSON"),
    ({chr(0xD800): 1}, ValueError, "did not round-trip through JSON"),  # a lone surrogate key
    ({"a": np.int64(3)}, TypeError, "not JSON serializable"),
    ([np.float32(1.5)], TypeError, "not JSON serializable"),
    ((object(),), TypeError, "not JSON serializable"),
    (_circular_list(), ValueError, "^Circular reference detected$"),
    (_circular_dict(), ValueError, "^Circular reference detected$"),
    ([{"a": _circular_list()}], ValueError, "^Circular reference detected$"),
    (_nested(200, 1), ValueError, "did not round-trip through JSON: recursion limit exceeded"),
]

# Each record type's `load_*` method and index attribute.
_READERS = {"annotations": ("load_anns", "anns"), "images": ("load_imgs", "imgs"), "categories": ("load_cats", "cats")}


def _typed(value):
    """Type and repr: tells 1 from 1.0 from True, and 0.0 from -0.0."""
    return type(value).__name__, repr(value)


def _with_custom(record, custom):
    """`tiny_dataset()` with `custom` on record id 1 of `record`, half the keys
    before its schema keys and half after."""
    ds = tiny_dataset()
    items = list(custom.items())
    half = len(items) // 2
    ds[record][0] = {**dict(items[:half]), **ds[record][0], **dict(items[half:])}
    return ds


def _raw_json(path):
    """A saved file with key order kept and every number as its literal text."""
    return json.loads(
        path.read_text(), object_pairs_hook=list, parse_int=lambda s: ("int", s), parse_float=lambda s: ("float", s)
    )


def _reads(coco, record, record_id=1):
    """Record `record_id` of `record` as `dataset`, the `load_*` method, and the index show it."""
    load, index = _READERS[record]
    return [
        next(r for r in coco.dataset[record] if r["id"] == record_id),
        getattr(coco, load)(record_id)[0],
        getattr(coco, index)[record_id],
    ]


def _views(ds, record, path):
    """Record id 1 of `record` from every read of it, and as `save` writes it."""
    coco = COCO(ds)
    coco.save(str(path))
    saved = next(r for r in dict(_raw_json(path))[record] if ("id", ("int", "1")) in r)
    return _reads(coco, record), saved


def _assert_matches_json_path(record, custom, tmp_path):
    """`custom` comes back from every view exactly as it does when the record
    is forced through json.dumps."""
    got, got_saved = _views(_with_custom(record, custom), record, tmp_path / "direct.json")
    want, want_saved = _views(_with_custom(record, {**custom, **_FORCE}), record, tmp_path / "json.json")
    for got_dict, want_dict in zip(got, want):
        want_dict = {k: v for k, v in want_dict.items() if k != FORCE_JSON_PATH}
        assert list(got_dict) == list(want_dict)  # the same keys, in the same order
        assert {k: _typed(v) for k, v in got_dict.items()} == {k: _typed(v) for k, v in want_dict.items()}
    assert got_saved == [pair for pair in want_saved if pair[0] != FORCE_JSON_PATH]
    return got[0]


def _random_scalars(rng):
    """Finite floats from random bit patterns, and ints across the i64 and
    u64 ranges."""
    bits = rng.integers(0, 2**64, size=4000, dtype=np.uint64, endpoint=False)
    values = [float(f) for f in bits.view(np.float64) if np.isfinite(f)]
    values += [int(i) for i in rng.integers(-(2**63), 2**63 - 1, size=1000, endpoint=True)]
    values += [2**63 + int(i) for i in rng.integers(0, 2**63 - 1, size=1000, endpoint=True)]
    return values


def _random_nested(rng, depth=0):
    """A random exact list, tuple, or str-keyed dict of FAST_SCALARS values."""
    scalars = list(FAST_SCALARS.values())
    kind = int(rng.integers(0, 9 if depth < 4 else 5))
    if kind < 5:
        return scalars[int(rng.integers(len(scalars)))]
    items = [_random_nested(rng, depth + 1) for _ in range(int(rng.integers(0, 4)))]
    if kind == 5:
        return items
    if kind == 6:
        return tuple(items)
    keys = rng.permutation(["b", "a", "zz", "é", "", "10", "9", "B"])
    return {str(k): v for k, v in zip(keys, items)}


class TestCustomScalarsMatchJsonPath:
    @pytest.mark.parametrize("record", list(_READERS))
    def test_scalars_match_json_path(self, record, tmp_path):
        out = _assert_matches_json_path(record, FAST_SCALARS, tmp_path)
        assert [k for k in out if k in FAST_SCALARS] == list(FAST_SCALARS)
        for key, value in FAST_SCALARS.items():
            assert _typed(out[key]) == _typed(json.loads(json.dumps(value))), key

    @pytest.mark.parametrize("record", list(_READERS))
    @pytest.mark.parametrize("key", list(JSON_PATH_VALUES))
    def test_other_values_match_json_path(self, record, key, tmp_path):
        # One per record, between scalars, so the record leaves the direct
        # path after converting some of them.
        _assert_matches_json_path(record, {"a": 1.0, key: JSON_PATH_VALUES[key], "z": "s"}, tmp_path)

    @pytest.mark.parametrize("record", list(_READERS))
    @pytest.mark.parametrize("key_type", [_HashOverride, _EqOverride])
    def test_str_subclass_key_matches_json_path(self, record, key_type, tmp_path):
        # Converted directly here; forced, the entry is converted first and
        # then handed to json.dumps with the forcing value.
        out = _assert_matches_json_path(record, {key_type("tag"): 1.0, "z": "s"}, tmp_path)
        assert out["tag"] == 1.0

    @pytest.mark.parametrize("key_type", [_HashOverride, _EqOverride])
    def test_json_path_gets_the_original_entries(self, key_type, dumps_calls):
        """A record that leaves the direct path partway hands json.dumps its
        custom entries in order, as the objects it holds. Looking the entries
        converted so far up again by name misses a key whose hash or equality
        is not `str`'s, and drops its value."""
        tag, later = key_type("tag"), {1: "x"}
        coco = COCO(_with_custom("annotations", {tag: 1.0, "later": later}))
        (call,) = dumps_calls
        assert [str(k) for k in call] == ["tag", "later"]
        (k0, v0), (_, v1) = call.items()
        assert type(k0) is key_type and k0 is tag and v0 == 1.0 and v1 is later
        ann = coco.load_anns(1)[0]
        assert ann["tag"] == 1.0 and ann["later"] == {"1": "x"}

    def test_each_value_takes_the_path_compared(self, dumps_calls):
        """The comparisons here are not vacuous. Every value they treat as
        converted directly makes no json.dumps call, each record carrying one
        declined value makes exactly one, and so does the forced record."""
        COCO(_with_custom("annotations", {**FAST_SCALARS, **NESTED_VALUES}))
        assert dumps_calls == []
        ds = tiny_dataset()
        ds["annotations"] = [
            {"id": i + 1, "image_id": 1, "category_id": 1, "a": 1.0, "v": v, "z": "s"}
            for i, v in enumerate(JSON_PATH_VALUES.values())
        ]
        COCO(ds)
        assert [call["v"] for call in dumps_calls] == list(JSON_PATH_VALUES.values())
        dumps_calls.clear()
        COCO(_with_custom("annotations", _FORCE))
        assert len(dumps_calls) == 1

    def test_random_floats_and_ints_match_json_path(self, tmp_path):
        values = _random_scalars(np.random.default_rng(0))

        def build(force):
            ds = tiny_dataset()
            extra = _FORCE if force else {}
            ds["annotations"] = [
                {"id": i + 1, "image_id": 1, "category_id": 1, "v": v, **extra} for i, v in enumerate(values)
            ]
            return ds

        direct, via_json = COCO(build(False)), COCO(build(True))
        got = [_typed(a["v"]) for a in direct.dataset["annotations"]]
        assert got == [_typed(a["v"]) for a in via_json.dataset["annotations"]]
        direct.save(str(tmp_path / "direct.json"))
        via_json.save(str(tmp_path / "json.json"))
        saved = [dict(a)["v"] for a in dict(_raw_json(tmp_path / "direct.json"))["annotations"]]
        assert saved == [dict(a)["v"] for a in dict(_raw_json(tmp_path / "json.json"))["annotations"]]

    @pytest.mark.parametrize("value, error, message", REJECTED_VALUES)
    @pytest.mark.parametrize("position", ["alone", "after_scalars", "before_scalars"])
    def test_rejected_values_still_raise(self, value, error, message, position):
        scalars = {"a": 1.0, "b": 2, "c": "s"}
        custom = {
            "alone": {"bad": value},
            "after_scalars": {**scalars, "bad": value},
            "before_scalars": {"bad": value, **scalars},
        }[position]
        with pytest.raises(error, match=message):
            COCO(_with_custom("annotations", custom))

    def test_update_anns_matches_json_path(self):
        def updated(force):
            coco = COCO(tiny_dataset())
            extra = _FORCE if force else {}
            coco.update_anns([{"id": 1, **FAST_SCALARS, **NESTED_VALUES, **extra}], create=True)
            out = coco.load_anns(1)[0]
            return {k: _typed(v) for k, v in out.items() if k != FORCE_JSON_PATH}, list(out)

        (got, got_order), (want, want_order) = updated(False), updated(True)
        assert got == want
        assert got_order == [k for k in want_order if k != FORCE_JSON_PATH]

    def test_update_anns_still_rejects_an_unknown_scalar_key(self):
        ds = tiny_dataset()
        ds["annotations"][0]["score_source"] = "model"
        coco = COCO(ds)
        coco.update_anns([{"id": 1, "score_source": "human"}])
        assert coco.load_anns(1)[0]["score_source"] == "human"
        with pytest.raises(KeyError, match="not an annotation field"):
            coco.update_anns([{"id": 1, "score_src": 1.0}])


# ---------------------------------------------------------------------------
# Custom values read back as json.loads gives them
#
# Every read (dataset, load_*, anns/imgs/cats) builds a record's custom values
# in Rust from the stored JSON. The tests above compare two ingest paths that
# share that read, so a bug in it would pass them; these check it against
# Python's json module instead, with the two ways serde_json's storage differs
# from json.loads applied by hand: a nested dict is sorted by key (no
# preserve_order feature; sorting UTF-8 bytes and sorting code points agree),
# and an int outside i64 and u64 is the nearest float.
# ---------------------------------------------------------------------------


def _exact(value):
    """`_typed` all the way down, keeping dict order and key types."""
    if type(value) is dict:
        return "dict", [(_typed(k), _exact(v)) for k, v in value.items()]
    if type(value) is list:
        return "list", [_exact(v) for v in value]
    return _typed(value)


def _serde_int(text):
    i = int(text)
    return i if -(2**63) <= i < 2**64 else float(i)


def _sorted_dicts(value):
    if type(value) is dict:
        return {k: _sorted_dicts(value[k]) for k in sorted(value)}
    if type(value) is list:
        return [_sorted_dicts(v) for v in value]
    return value


def _expected(value):
    """What a custom value reads back as."""
    return _exact(_sorted_dicts(json.loads(json.dumps(value), parse_int=_serde_int)))


def _load(ds, source, tmp_path):
    if source == "dict":
        return COCO(ds)
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(ds))
    return COCO(str(path))


_READ_BACK_CASES = {
    "direct": {**FAST_SCALARS, **NESTED_VALUES},
    # One declined value sends a COCO(dict) record through json.dumps.
    "json_dumps": {**FAST_SCALARS, **NESTED_VALUES, **JSON_PATH_VALUES},
}


class TestCustomValuesReadBackAsJsonLoads:
    @pytest.mark.parametrize("source", ["dict", "file"])
    @pytest.mark.parametrize("record", list(_READERS))
    @pytest.mark.parametrize("case", list(_READ_BACK_CASES))
    def test_every_value_kind(self, case, record, source, tmp_path):
        custom = _READ_BACK_CASES[case]
        coco = _load(_with_custom(record, custom), source, tmp_path)
        want = [(k, _expected(v)) for k, v in custom.items()]
        for out in _reads(coco, record):
            assert list(out)[-len(custom) :] == list(custom)  # after the schema keys, in file order
            assert [(k, _exact(out[k])) for k in custom] == want

    @pytest.mark.parametrize("source", ["dict", "file"])
    def test_random_values(self, source, tmp_path):
        rng = np.random.default_rng(2)
        values = _random_scalars(rng)
        values += [-0.0, 0.0, 1.0, -1.0, 2.0**53, 2.0**63, 2.0**64, 1e16, 5e-324, sys.float_info.max]
        bounds = [0, 1, 2**53, 2**53 + 1, 2**63 - 1, 2**63, 2**64 - 1, 2**64, 2**64 + 1]
        bounds += [2**64 + 2**11, 2**64 + 3 * 2**11]  # halfway between two floats: ties to even
        values += bounds + [-b for b in bounds]
        # Wider than u64, so stored as the nearest float.
        values += [int(rng.integers(1, 2**62)) << int(s) for s in rng.integers(64, 960, size=500)]
        values += [-int(rng.integers(1, 2**62)) << int(s) for s in rng.integers(64, 960, size=500)]
        values += [_random_nested(rng) for _ in range(500)]
        ds = tiny_dataset()
        ds["annotations"] = [{"id": i + 1, "image_id": 1, "category_id": 1, "v": v} for i, v in enumerate(values)]
        coco = _load(ds, source, tmp_path)
        want = [_expected(v) for v in values]
        assert [_exact(a["v"]) for a in coco.dataset["annotations"]] == want
        assert [_exact(a["v"]) for a in coco.load_anns(list(range(1, len(values) + 1)))] == want
        anns = coco.anns
        assert [_exact(anns[i + 1]["v"]) for i in range(len(values))] == want


def _torchmetrics_dataset(predictions):
    """A dataset shaped the way TorchMetrics' ``_get_coco_format`` builds one for
    a bbox+segm run: ``area_bbox`` and ``area_segm`` on every annotation, RLE
    ``counts`` as bytes, ``size`` a tuple, and ``score`` on predictions."""
    rng = np.random.default_rng(7)
    images, anns = [], []
    for image_id in range(6):
        images.append({"id": image_id})
        for _ in range(4):
            x, y = (int(v) for v in rng.integers(0, 20, size=2))
            w, h = (int(v) for v in rng.integers(4, 12, size=2))
            if predictions:  # jitter, so matches land at several IoUs
                x, y = x + int(rng.integers(0, 3)), y + int(rng.integers(0, 3))
            m = np.zeros((32, 32), np.uint8)
            m[y : y + h, x : x + w] = 1
            rle = mask.encode(np.asfortranarray(m))
            assert isinstance(rle["counts"], bytes)
            bbox = [float(v) for v in mask.toBbox(rle)]
            ann = {
                "id": len(anns) + 1,
                "image_id": image_id,
                "area": int(mask.area(rle)),
                "category_id": 1 + len(anns) % 3,
                "iscrowd": 0,
                "area_bbox": bbox[2] * bbox[3],
                "area_segm": int(mask.area(rle)),
                "bbox": bbox,
                "segmentation": {"size": tuple(rle["size"]), "counts": rle["counts"]},
            }
            if predictions:
                ann["score"] = float(rng.random())
            anns.append(ann)
    categories = [{"id": c, "name": str(c)} for c in (1, 2, 3)]
    return {"images": images, "annotations": anns, "categories": categories}


def _strip(ds, keys):
    return {**ds, "annotations": [{k: v for k, v in a.items() if k not in keys} for a in ds["annotations"]]}


class TestTorchMetricsShapedAnnotations:
    """TorchMetrics (and RF-DETR through it) adds two scalar custom keys to every
    annotation; they are the reason the direct path exists."""

    def test_custom_keys_round_trip(self, tmp_path):
        dt = _torchmetrics_dataset(predictions=True)
        coco = COCO(dt)
        coco.save(str(tmp_path / "dt.json"))
        saved = dict(_raw_json(tmp_path / "dt.json"))["annotations"]
        for src, out, raw in zip(dt["annotations"], coco.load_anns([a["id"] for a in dt["annotations"]]), saved):
            assert _typed(out["area_bbox"]) == _typed(src["area_bbox"])
            assert _typed(out["area_segm"]) == _typed(src["area_segm"])
            assert dict(raw)["area_bbox"][0] == "float" and dict(raw)["area_segm"][0] == "int"
        res = COCO(_torchmetrics_dataset(predictions=False)).load_res(dt["annotations"])
        assert _typed(res.load_anns(1)[0]["area_bbox"]) == _typed(dt["annotations"][0]["area_bbox"])

    @pytest.mark.parametrize("iou_type", ["bbox", "segm"])
    def test_evaluates_as_without_them(self, iou_type, capsys):
        custom = {"area_bbox", "area_segm"}

        def stats(gt, dt):
            ev = COCOeval(COCO(gt), COCO(dt), iou_type)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            return list(ev.stats)

        gt, dt = _torchmetrics_dataset(predictions=False), _torchmetrics_dataset(predictions=True)
        got = stats(gt, dt)
        assert got == stats(_strip(gt, custom), _strip(dt, custom))
        assert max(got) > 0


# ---------------------------------------------------------------------------
# Every known dict key survives decode and read-back
#
# convert.rs names each schema key with a string literal: through
# pyo3::intern! in the decode macros (opt!/req!/opt_with!/req_with!), the
# get_item calls in py_to_annotation/py_to_segmentation/py_to_rle, and the
# record builders (annotation_to_py and the rest), and as a match arm in
# set_ann_field. A typo in one breaks silently: a required key (id, image_id)
# raises "dict missing '<real name>'", an optional key on the way in vanishes
# or lands among the custom keys, and one on the way out comes back misspelled.
# Each record below carries every schema key of its type, and some of other
# record types' schema keys, which are custom keys on it.
# ---------------------------------------------------------------------------


def _schema_records():
    """One record of each type, with every schema key of that type in the
    order the binding writes them."""
    return {
        "images": {
            "id": 9,
            "file_name": "x.jpg",
            "height": 100,
            "width": 100,
            "license": 3,
            "coco_url": "http://a",
            "flickr_url": "http://b",
            "date_captured": "2020-01-01",
            "neg_category_ids": [5, 6],
            "not_exhaustive_category_ids": [7],
        },
        "annotations": {
            "id": 7,
            "image_id": 9,
            "category_id": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "area": 12.5,
            "segmentation": {"size": [10, 10], "counts": [100]},
            "iscrowd": 1,
            "keypoints": [1.0, 2.0, 2.0],
            "num_keypoints": 1,
            "obb": [5.0, 5.0, 2.0, 2.0, 0.3],
            "score": 0.75,
            "is_group_of": 1,
        },
        "categories": {
            "id": 1,
            "name": "person",
            "supercategory": "animal",
            "skeleton": [[0, 1], [1, 2]],
            "keypoints": ["nose", "eye"],
            "frequency": "f",
        },
    }


def _every_key_dataset():
    custom = {
        "images": {
            "bbox": [1, 2, 3, 4],
            "score": 0.5,
            "name": "an image",
            "skeleton": {"custom": True},
            "weather": "rainy",
        },
        "annotations": {
            "file_name": "an annotation",
            "height": 7,
            "frequency": "r",
            "supercategory": None,
            "attributes": {"occluded": False, "tags": ["a", "b"]},
        },
        "categories": {"iscrowd": 1, "width": 3.0, "coco_url": "http://c", "taxonomy_id": 42},
    }
    return {record: [{**fields, **custom[record]}] for record, fields in _schema_records().items()}


class TestKnownKeysRoundTrip:
    @pytest.mark.parametrize("record", list(_READERS))
    def test_every_field_survives(self, record):
        ds = _every_key_dataset()
        src = ds[record][0]
        want = _exact({**src, "is_group_of": True} if record == "annotations" else src)
        for out in _reads(COCO(ds), record, src["id"]):
            assert _exact(out) == want


# ---------------------------------------------------------------------------
# A key reads the same whichever str object spells it
#
# A dict literal, a record hotcoco returns, and json.loads output spell the same
# keys as equal strings that are not always the same objects. These tests
# compare the records each spelling gives, so a fast path that names a key by
# object identity instead of by its text cannot come back unnoticed.
# ---------------------------------------------------------------------------


class TestKeyObjectsDoNotMatter:
    def test_loaded_keys_read_like_literal_keys(self, tmp_path):
        literal = _every_key_dataset()
        loaded = json.loads(json.dumps(literal))
        views = []
        for ds, name in ((literal, "literal.json"), (loaded, "loaded.json")):
            coco = COCO(ds)
            coco.save(str(tmp_path / name))
            views.append((_exact(coco.dataset), _raw_json(tmp_path / name)))
        assert views[0] == views[1]

    def test_read_back_records_round_trip(self, tmp_path):
        coco = COCO(_every_key_dataset())
        want = _exact(coco.dataset)
        again = COCO(coco.dataset)
        assert _exact(again.dataset) == want
        again.update_anns(again.load_anns(again.get_ann_ids()))
        assert _exact(again.dataset) == want
        coco.save(str(tmp_path / "first.json"))
        again.save(str(tmp_path / "again.json"))
        assert _raw_json(tmp_path / "first.json") == _raw_json(tmp_path / "again.json")

    def test_update_anns_loaded_keys_read_like_literal_keys(self):
        def updated(ann):
            coco = COCO(_every_key_dataset())
            coco.update_anns([ann], create=True)
            return _exact(coco.load_anns(7)[0])

        literal = {"id": 7, "score": 0.25, "iscrowd": 0, "bbox": [0.0, 0.0, 1.0, 1.0], "name": "renamed", "new": [1]}
        assert updated(literal) == updated(json.loads(json.dumps(literal)))

    @pytest.mark.parametrize("record", list(_READERS))
    def test_key_with_no_text_raises(self, record):
        """Pins a known difference from pycocotools: a lone-surrogate key raises.

        pycocotools keeps such a key, and `json.dump` writes it back escaped. hotcoco
        stores keys as UTF-8 Rust strings, and a lone surrogate has no UTF-8 form, so
        the key raises instead of being read as some other key. hotcoco 1.1 raised the
        same error. docs/getting-started/migration.md lists it under Known differences.
        """
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            COCO(_with_custom(record, {chr(0xD800): 1}))


# ---------------------------------------------------------------------------
# annToRLE returns pycocotools format
# ---------------------------------------------------------------------------


class TestAnnToRle:
    def test_ann_to_rle_pycocotools_format(self):
        ds = tiny_dataset()
        ds["annotations"][0]["segmentation"] = [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]]
        coco = COCO(ds)
        rle = coco.ann_to_rle(coco.anns[1])
        assert set(rle) == {"size", "counts"}
        assert rle["size"] == [100, 100]
        assert isinstance(rle["counts"], bytes)
        # Feeds straight back into the mask module, like pycocotools
        assert mask.decode(rle).sum() == mask.area(rle)
        assert coco.annToRLE(coco.anns[1]) == rle


# ---------------------------------------------------------------------------
# ev.stats: [] before summarize, ndarray after — and params copy docs hold
# ---------------------------------------------------------------------------


class TestStatsSemantics:
    def test_stats_empty_list_before_summarize(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        assert ev.stats == []

    def test_stats_ndarray_after_summarize(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.run()
        stats = ev.stats
        assert isinstance(stats, np.ndarray)
        assert stats.dtype == np.float64
        assert stats.shape == (12,)
        # The pycocotools idiom this shape exists for:
        assert stats[0] >= 0.0

    def test_params_max_dets_is_a_list(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        assert ev.params.maxDets == [1, 10, 100]
        assert isinstance(ev.params.maxDets, list)
        # Whole-attribute assignment is the documented mutation path.
        ev.params.maxDets = [1, 10, 100, 200]
        assert ev.params.maxDets == [1, 10, 100, 200]


# ---------------------------------------------------------------------------
# Guards emit real Python warnings, not fd-2 writes
# ---------------------------------------------------------------------------


class TestGuardsWarn:
    def test_accumulate_before_evaluate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="accumulate"):
            ev.accumulate()

    def test_summarize_before_accumulate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="summarize"):
            ev.summarize()

    def test_f_scores_before_accumulate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="f_scores"):
            assert ev.f_scores() == {}

    def test_no_warning_on_correct_order(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ev.run()


# ---------------------------------------------------------------------------
# Error types: converters route through to_pyerr
# ---------------------------------------------------------------------------


class TestErrorTypes:
    def test_from_voc_missing_dir_is_oserror(self):
        with pytest.raises(OSError):
            COCO.from_voc("/nonexistent/voc-dir")

    def test_from_yolo_missing_data_yaml_is_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="data.yaml"):
            COCO.from_yolo(str(tmp_path))

    def test_from_cvat_malformed_xml_is_value_error(self, tmp_path):
        bad = tmp_path / "annotations.xml"
        bad.write_text("<annotations><image name='a.jpg'")
        with pytest.raises(ValueError):
            COCO.from_cvat(str(bad))

    def test_compare_mismatched_params_is_value_error(self):
        gt = COCO(tiny_dataset())
        ev_a = COCOeval(gt, tiny_dt(gt), "bbox")
        ev_a.evaluate()
        ev_b = COCOeval(gt, tiny_dt(gt), "bbox")
        ev_b.params.iouThrs = [0.5]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # deliberate non-reference config
            ev_b.evaluate()
        with pytest.raises(ValueError):
            hotcoco.compare(ev_a, ev_b)


# ---------------------------------------------------------------------------
# mask module round-trip and dict-in/dict-out
# ---------------------------------------------------------------------------


class TestMaskSurface:
    def test_encode_decode_round_trip(self):
        m = np.zeros((10, 12), dtype=np.uint8, order="F")
        m[2:5, 3:7] = 1
        rle = mask.encode(m)
        assert isinstance(rle["counts"], bytes)
        assert (mask.decode(rle) == m).all()

    def test_encode_3d_round_trip(self):
        m = np.zeros((10, 12, 2), dtype=np.uint8, order="F")
        m[2:5, 3:7, 0] = 1
        m[6:9, 1:4, 1] = 1
        rles = mask.encode(m)
        assert isinstance(rles, list) and len(rles) == 2
        assert (mask.decode(rles) == m).all()

    def test_fr_py_objects_dict_in_dict_out(self):
        uncompressed = {"size": [10, 10], "counts": [50, 10, 40]}
        out = mask.frPyObjects(uncompressed, 10, 10)
        assert isinstance(out, dict), "dict input must give dict output (pycocotools parity)"
        outs = mask.frPyObjects([uncompressed], 10, 10)
        assert isinstance(outs, list) and len(outs) == 1


# ---------------------------------------------------------------------------
# LVIS drop-in objects construct the documented types
# ---------------------------------------------------------------------------


class TestLvisDropIn:
    def test_lviseval_returns_cocoeval(self):
        gt = COCO(tiny_dataset())
        ev = LVISeval(gt, tiny_dt(gt), "bbox")
        assert isinstance(ev, COCOeval)

    def test_lvisresults_returns_coco(self):
        gt = COCO(tiny_dataset())
        res = LVISResults(gt, [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.5}])
        assert isinstance(res, COCO)


# ---------------------------------------------------------------------------
# load_warnings
# ---------------------------------------------------------------------------


class TestLoadWarnings:
    def test_clean_load_has_no_warnings(self):
        assert COCO(tiny_dataset()).load_warnings == []

    def test_duplicate_ann_ids_are_reported(self):
        ds = tiny_dataset()
        ds["annotations"][1]["id"] = 1  # collide with the first
        coco = COCO(ds)
        assert any("annotation" in w for w in coco.load_warnings)


# ---------------------------------------------------------------------------
# Issue #5: findings from adopting hotcoco through TorchMetrics
# ---------------------------------------------------------------------------


def _one_box_dataset():
    return {
        "images": [{"id": 0, "height": 10, "width": 10}],
        "categories": [{"id": 1, "name": "1"}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0}],
    }


def _square_mask(dtype=np.uint8):
    m = np.zeros((10, 10), dtype=dtype)
    m[2:6, 2:6] = 1
    return np.asfortranarray(m)


class TestIssue5BytesCounts:
    """The reported symptom: identical masks scoring segm AP 0.0 when `counts` is bytes.

    The round-trip itself is pinned by `tests/test_parity.py`; this is the
    evaluation-level check that nothing else covers.
    """

    def test_segm_ap_is_one_for_identical_masks(self):
        rle = mask.encode(_square_mask())
        gt_ds, dt_ds = _one_box_dataset(), _one_box_dataset()
        gt_ds["annotations"][0]["segmentation"] = dict(rle)
        dt_ds["annotations"][0]["segmentation"] = dict(rle)
        dt_ds["annotations"][0]["score"] = 0.9
        ev = COCOeval(COCO(gt_ds), COCO(dt_ds), iou_type="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # A lone true positive's AP is what ``metrics.average_precision`` says
        # it is (pycocotools' guard term puts it an ulp under 1.0), to within
        # the summary mean's rounding.
        lone_tp = hotcoco.metrics.average_precision([1.0], [True], 1)
        assert ev.stats[0] == pytest.approx(lone_tp, abs=np.finfo(float).eps)


class TestIssue5MaskEncodeDtype:
    """The issue #5 cases end to end. `TestEncodeScan` in
    `tests/test_mask_parity.py` checks every layout and one-byte dtype against
    pycocotools."""

    def test_sliced_view_matches_contiguous(self):
        f_order = _square_mask()
        wide = np.zeros((10, 20), np.uint8)
        wide[:, 5:15] = f_order
        assert mask.encode(wide[:, 5:15]) == mask.encode(f_order)

    def test_int8_is_nonzero_foreground(self):
        m = _square_mask(np.int8)
        m[2, 2] = -1
        assert mask.encode(m) == mask.encode(_square_mask())


class TestIssue5MissingCategoryName:
    def test_missing_name_gets_placeholder_and_warns(self):
        ds = tiny_dataset()
        ds["categories"][0] = {"id": 1}
        coco = COCO(ds)
        assert coco.loadCats(1)[0]["name"] == "cat_1"
        assert any("without a name" in w for w in coco.load_warnings)


class TestIssue5SummarizeOutput:
    """`summarize()` prints through `sys.stdout`, so Python-level redirection works,
    and its parameter warnings are Python warnings — emitted once, not also on fd 2."""

    @staticmethod
    def _evaluated(max_dets=None):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        if max_dets:
            ev.params.maxDets = max_dets
        ev.evaluate()
        ev.accumulate()
        return ev

    def test_table_goes_through_sys_stdout(self, capsys):
        # capsys swaps sys.stdout, exactly as contextlib.redirect_stdout does.
        self._evaluated().summarize()
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 12
        assert out[0].startswith(" Average Precision (AP) @[ IoU=0.50:0.95")

    def test_param_warning_is_a_python_warning_only(self, capfd):
        ev = self._evaluated(max_dets=[1, 10, 500])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ev.summarize()
        assert [w for w in caught if "max_dets" in str(w.message)]
        assert "max_dets" not in capfd.readouterr().err


# ---------------------------------------------------------------------------
# Found by tests/fuzz_dropin.py: spellings pycocotools accepts
# ---------------------------------------------------------------------------


def _kp_dataset(num_keypoints=True, as_array=False):
    kps = [0.0] * 51
    for k in range(5):
        kps[k * 3 : k * 3 + 3] = [10.0 + k, 10.0 + k, 2.0]
    ann = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "area": 400, "iscrowd": 0}
    ann["keypoints"] = np.asarray(kps).reshape(-1, 3) if as_array else kps
    if num_keypoints:
        ann["num_keypoints"] = 5
    return {
        "images": [{"id": 1, "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "person", "keypoints": [f"k{i}" for i in range(17)], "skeleton": []}],
        "annotations": [ann],
    }


def _kp_stats(gt_ds):
    gt = COCO(gt_ds)
    det = dict(gt_ds["annotations"][0])
    det["score"] = 0.9
    det["keypoints"] = np.asarray(det["keypoints"]).ravel().tolist()
    dt = gt.loadRes([det])
    ev = COCOeval(gt, dt, "keypoints")
    ev.evaluate()
    ev.accumulate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.summarize()
    return list(ev.stats)


class TestFuzzDropinFindings:
    """Dict-path spellings only; the `num_keypoints` rule itself is pinned in Rust."""

    def test_keypoints_as_nx3_array(self):
        assert _kp_stats(_kp_dataset(as_array=True)) == _kp_stats(_kp_dataset())

    def test_integral_float_ids_load(self):
        ds = tiny_dataset()
        for img in ds["images"]:
            img["id"] = float(img["id"])
        for ann in ds["annotations"]:
            ann["id"], ann["image_id"], ann["category_id"] = (
                float(ann["id"]),
                float(ann["image_id"]),
                float(ann["category_id"]),
            )
            ann["iscrowd"] = 0.0
        for cat in ds["categories"]:
            cat["id"] = float(cat["id"])
        coco = COCO(ds)
        assert coco.getImgIds() == [1, 2]
        assert coco.loadAnns(1)[0]["category_id"] == 1
        assert coco.loadRes(
            [{"image_id": 1.0, "category_id": 1.0, "bbox": [1, 1, 2, 2], "score": 0.5}]
        ).getAnnIds() == [1]

    def test_fractional_id_is_rejected(self):
        ds = tiny_dataset()
        ds["images"][0]["id"] = 1.5
        with pytest.raises(TypeError, match="non-negative integer, got 1.5"):
            COCO(ds)

    def test_out_of_range_int_names_the_overflow(self):
        ds = tiny_dataset()
        ds["images"][0]["height"] = 2**40
        with pytest.raises(OverflowError, match="out of range"):
            COCO(ds)

    # convert.rs reads an exact bool or an int in the i64 range first, and
    # everything else by the generic path; the two must agree everywhere.
    FLAGS = [True, False, 0, 1, 2, -1, -(2**63), 2**63 - 1, 2**63, 2**64, -(2**63) - 1, 0.0, 1.0, -0.0]
    FLAGS += [np.bool_(True), np.bool_(False), np.int64(1), np.int64(0), np.float64(1.0), _IntSub(0), _IntSub(3)]

    @pytest.mark.parametrize("value", FLAGS)
    def test_flags_share_one_reader(self, value):
        ds = tiny_dataset()
        ds["annotations"][0]["iscrowd"] = value
        ds["annotations"][0]["is_group_of"] = value
        ann = COCO(ds).loadAnns(1)[0]
        assert ann["iscrowd"] == int(bool(value))
        assert ann["is_group_of"] is bool(value)
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.params.useCats = value
        assert ev.params.useCats is bool(value)
        ev.params.use_cats = value
        assert ev.params.use_cats is bool(value)
        assert gt.getAnnIds(iscrowd=value) == ([] if value else [1, 2])

    # One reader is enough: test_flags_share_one_reader pins that every flag
    # goes through the same one.
    @pytest.mark.parametrize("value", [0.5, -0.5, float("nan"), float("inf"), np.float32(0.5), 2**1024, "0", None, [1]])
    def test_non_flags_are_rejected(self, value):
        ds = tiny_dataset()
        ds["annotations"][0]["iscrowd"] = value
        with pytest.raises(TypeError, match=f"^expected a bool or 0/1 flag, got {re.escape(repr(value))}"):
            COCO(ds)
