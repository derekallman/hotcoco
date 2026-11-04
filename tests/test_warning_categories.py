"""Every warning hotcoco emits carries a category a filter can name.

One row per call site that emits a warning. Each row is held to two things: a
filter on its own category catches it, as that exact class and not a sibling;
and its message starts with ``hotcoco: ``. A plain ``UserWarning`` filter written
before the categories existed still catches every one, because each category is
a ``UserWarning`` subclass and a filter matches by ``issubclass``.
``test_every_call_site_has_a_row`` counts the call sites in the bindings and the
Python package, so a new warning without a row fails here. Neither a row nor a
call site names the base, ``HotcocoWarning``: nothing emits it, so a filter on
one category never has to catch another kind too.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter

import hotcoco
import numpy as np
import pytest
from helpers import WORKSPACE
from hotcoco import (
    COCO,
    CallOrderWarning,
    COCOeval,
    CocoEvaluator,
    ConversionWarning,
    DatasetWarning,
    HotcocoWarning,
    ReferenceDeviationWarning,
    panoptic,
    semantic,
)

PYO3_SRC = WORKSPACE / "crates" / "hotcoco-pyo3" / "src"
LIB_RS = "crates/hotcoco-pyo3/src/lib.rs"
LIVE_RS = "crates/hotcoco-pyo3/src/live.rs"
PANOPTIC_RS = "crates/hotcoco-pyo3/src/panoptic.rs"
SEMANTIC_RS = "crates/hotcoco-pyo3/src/semantic.rs"
INTEGRATIONS_PY = "python/hotcoco/integrations.py"

IMAGES = [{"id": 1, "width": 100, "height": 100}]
CATS = [{"id": 1, "name": "thing"}]  # no `isthing`: a panoptic reference deviation
GT = {
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [10, 10, 40, 40],
    "area": 1600,
    "iscrowd": 0,
    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
}
DT = {"image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "score": 0.9}

CVAT_WITH_A_POLYLINE = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels><label><name>thing</name></label></labels></task></meta>
  <image id="0" name="img.jpg" width="100" height="100">
    <box label="thing" xtl="10" ytl="10" xbr="50" ybr="50" occluded="0"/>
    <polyline label="thing" points="10,10;50,50" occluded="0"/>
  </image>
</annotations>
"""


def _gt():
    return COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})


def _ev(**params):
    gt = _gt()
    ev = COCOeval(gt, gt.load_res([DT]), "bbox")
    for name, value in params.items():
        setattr(ev.params, name, value)
    return ev


def _accumulated(**params):
    ev = _ev(**params)
    ev.evaluate()
    ev.accumulate()
    return ev


def _duplicate_ids(_):
    COCO({"images": IMAGES, "annotations": [GT, dict(GT)], "categories": CATS})


def _edit_makes_duplicate_ids(_):
    """The read-back of an edited `dataset` warns about the problem the edit made."""
    coco = COCO({"images": IMAGES, "annotations": [GT, {**GT, "id": GT["id"] + 1}], "categories": CATS})
    coco.dataset["annotations"][1]["id"] = GT["id"]
    coco.getAnnIds()


def _cvat_import(tmp_path):
    path = tmp_path / "annotations.xml"
    path.write_text(CVAT_WITH_A_POLYLINE)
    COCO.from_cvat(str(path))


# (trigger, category, message pattern, the file that emits it). A trigger takes
# `tmp_path` so the CVAT row can write its file.
SITES = [
    pytest.param(_duplicate_ids, DatasetWarning, "duplicate annotation id", LIB_RS, id="loader-duplicate-ids"),
    pytest.param(
        _edit_makes_duplicate_ids, DatasetWarning, "duplicate annotation id", LIVE_RS, id="read-back-duplicate-ids"
    ),
    pytest.param(_cvat_import, ConversionWarning, "skipped 1 unsupported", LIB_RS, id="cvat-skipped-shapes"),
    pytest.param(
        lambda _: _ev().accumulate(), CallOrderWarning, r"accumulate\(\) called before", LIB_RS, id="accumulate"
    ),
    pytest.param(lambda _: _ev().summarize(), CallOrderWarning, r"summarize\(\) called before", LIB_RS, id="summarize"),
    pytest.param(
        lambda _: _ev().summary_lines(),
        CallOrderWarning,
        r"summary_lines\(\) called before",
        LIB_RS,
        id="summary-lines",
    ),
    pytest.param(lambda _: _ev().print_results(), CallOrderWarning, "No results to print", LIB_RS, id="print-results"),
    pytest.param(lambda _: _ev().f_scores(), CallOrderWarning, r"f_scores\(\) called before", LIB_RS, id="f-scores"),
    pytest.param(
        lambda _: _accumulated(maxDets=[1, 10, 500]).summarize(),
        ReferenceDeviationWarning,
        "^hotcoco: max_dets differ from expected",
        LIB_RS,
        id="detection-deviation",
    ),
    pytest.param(
        lambda _: panoptic.PanopticEval(_gt(), _gt()).summarize(),
        CallOrderWarning,
        r"summarize\(\) called before evaluate",
        PANOPTIC_RS,
        id="panoptic-summarize",
    ),
    pytest.param(
        lambda _: panoptic.PanopticEval(_gt(), _gt()).run(),
        ReferenceDeviationWarning,
        "isthing",
        PANOPTIC_RS,
        id="panoptic-deviation",
    ),
    pytest.param(
        lambda _: panoptic.PanopticEval(_gt(), _gt()).summary_lines(),
        CallOrderWarning,
        r"summary_lines\(\) called before evaluate",
        PANOPTIC_RS,
        id="panoptic-summary-lines",
    ),
    pytest.param(
        lambda _: semantic.SemanticEval(2).summarize(),
        CallOrderWarning,
        r"summarize\(\) called before update",
        SEMANTIC_RS,
        id="semantic-summarize",
    ),
    pytest.param(
        lambda _: semantic.SemanticEval(2).summary_lines(),
        CallOrderWarning,
        r"summary_lines\(\) called before update",
        SEMANTIC_RS,
        id="semantic-summary-lines",
    ),
    pytest.param(
        lambda _: CocoEvaluator(_gt(), ["bbox"]).accumulate(),
        CallOrderWarning,
        "no results for iou_type='bbox'",
        INTEGRATIONS_PY,
        id="coco-evaluator-accumulate",
    ),
    pytest.param(
        lambda _: CocoEvaluator(_gt(), ["bbox"]).summarize(),
        CallOrderWarning,
        "no evaluation for iou_type='bbox'",
        INTEGRATIONS_PY,
        id="coco-evaluator-summarize",
    ),
]

# Call sites no input reaches today, so no row can trigger them.
# `SemanticEval.reference_deviations()` is always empty; when it gains an entry,
# `test_semantic_has_no_deviation_yet` fails, and that entry gets a row in SITES.
UNREACHABLE = Counter({SEMANTIC_RS: 1})


@pytest.mark.parametrize(("trigger", "category", "match", "_file"), SITES)
def test_a_filter_on_the_category_catches_exactly_it(tmp_path, trigger, category, match, _file):
    with warnings.catch_warnings():
        warnings.simplefilter("error", category)
        with pytest.raises(category, match=match) as caught:
            trigger(tmp_path)
    assert type(caught.value) is category
    # A downstream filter on `^hotcoco:` must match every hotcoco warning.
    assert str(caught.value).startswith("hotcoco: ")


def test_a_user_warning_filter_still_catches_it():
    """One emitted warning through a plain ``UserWarning`` filter. The rest follow from
    ``test_the_categories_nest_under_user_warning``: a filter matches by ``issubclass``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match=r"accumulate\(\) called before"):
            _ev().accumulate()


# Anything that raises a Python warning without going through the bindings' `warn`
# by that name, which is the only place a category is chosen.
# A `"warnings"` or `"warn"` string is Python's warnings module reached by name, in any
# spelling: `import(...)`, `intern!(...)`, `call_method1("warn", ...)`.
RAW_WARNING = re.compile(r"PyErr::warn\w*|PyErr_Warn\w*|\bPy\w*Warning\b|\"warn(?:ings|_explicit)?\"|\bwarn\s+as\b")


def _rust_code(text):
    """The source without its comment lines and `#[doc = "..."]` text, so prose about
    warnings cannot count as a call.

    Whole comment lines only: a `//` inside a string literal must not cut the code
    after it.
    """
    text = re.sub(r'#\[doc\s*=\s*"(?:[^"\\]|\\.)*"\]', "", text)
    text = re.sub(r'(?s)#\[doc\s*=\s*r(#*)".*?"\1\]', "", text)
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    # The base a category is created on names a warning type without raising one.
    text = re.sub(r"create_exception!\(\s*\w+\s*,\s*\w+\s*,\s*[\w:]+", "", text)
    # A trailing comment, told from a `//` in a string by having no quote after it.
    return re.sub(r'(?m)//[^"\n]*$', "", text)


def _warn_classes(text):
    """The class each call of the bindings' `warn` names, however qualified
    (`crate::warn::<…>(` too), `""` for a call that names none; not its definition or
    the `PyErr::warn(` inside it."""
    calls = re.findall(r"(?<!fn )(?<!PyErr::)\bwarn(?:\s*::\s*<\s*([\w:]+)\s*>)?\s*\(", _rust_code(text))
    return [name.rpartition("::")[2] for name in calls]


def _warn_calls(text):
    return len(_warn_classes(text))


def _non_leaf_warnings(text):
    """`warn` calls naming a class that is not a leaf category: the base, or none."""
    leaves = {category.__name__ for category in LEAVES}
    return [name or "no class" for name in _warn_classes(text) if name not in leaves]


def _rust_sources():
    """Each binding source file's workspace-relative name and text."""
    for path in sorted(PYO3_SRC.rglob("*.rs")):
        yield path.relative_to(WORKSPACE).as_posix(), path.read_text()


def _raw_warnings(text):
    return [m.group() for m in RAW_WARNING.finditer(_rust_code(text))]


def _py_warn_calls(text):
    """Calls of `warnings.warn` or `warn_explicit`, qualified or imported bare."""
    return len(re.findall(r"\bwarn(?:_explicit)?\(", text))


def _py_raw_warnings(text):
    """An alias, or `warn` reached by name, hides the calls from `_py_warn_calls`."""
    return re.findall(
        r"\bfrom\s+warnings\s+import\b.*\bas\b|\bimport\s+warnings\s+as\b|[\"']warn(?:_explicit)?[\"']", text
    )


def test_the_guard_sees_each_spelling_a_review_found():
    assert _warn_classes('let s = "https://x"; crate::warn::<ConversionWarning>(py, &s)?;') == ["ConversionWarning"]
    assert _non_leaf_warnings('warn::<HotcocoWarning>(py, "x")?;') == ["HotcocoWarning"]
    assert _non_leaf_warnings('crate::warn::<crate::HotcocoWarning>(py, "x")?;') == ["HotcocoWarning"]
    assert _non_leaf_warnings('warn::<CallOrderWarning>(py, "x")?;') == []
    assert _warn_calls("pub(crate) fn warn<W: pyo3::PyTypeInfo>(py: Python<'_>, msg: &str) -> PyResult<()> {") == 0
    assert _raw_warnings("use crate::warn as emit;") == ["warn as"]
    assert _raw_warnings('PyErr::warn(py, &py.get_type::<PyUserWarning>(), c"x", 1)?;') == [
        "PyErr::warn",
        "PyUserWarning",
    ]
    assert _warn_calls("    /// Routes through `PyErr::warn(`, unlike `warn(`.") == 0
    assert _raw_warnings('PyModule::import(py, "warnings")?.call_method1("warn", (msg,))?;') == ['"warnings"', '"warn"']
    assert _raw_warnings('let w = intern!(py, "warnings");') == ['"warnings"']
    doc = '#[doc = "Emitted through ``warnings.warn(`` as a ``PyUserWarning``,\nnot \\"warn(\\"."]\nfn f() {}'
    assert _warn_calls(doc) == 0 and _raw_warnings(doc) == []
    assert _py_warn_calls('warnings.warn_explicit("hotcoco: x", ConversionWarning, "f.py", 1)') == 1
    assert _py_raw_warnings("from warnings import warn as _w") == ["from warnings import warn as"]
    assert _py_raw_warnings("import warnings as w") == ["import warnings as"]
    assert _py_raw_warnings("getattr(warnings, 'warn')(msg)") == ["'warn'"]
    created = (
        "pyo3::create_exception!(\n"
        "    hotcoco,\n"
        "    HotcocoWarning,\n"
        "    pyo3::exceptions::PyUserWarning,\n"
        '    "doc"\n'
        ");"
    )
    assert _raw_warnings(created) == []
    assert _raw_warnings("let w = py.get_type::<pyo3::exceptions::PyUserWarning>();") == ["PyUserWarning"]
    assert _warn_calls("    evaluate(py)?; // unlike warn(") == 0
    raw_doc = '#[doc = r#"Emitted through ``warnings.warn(`` as a ``PyUserWarning``."#]\nfn f() {}'
    assert _warn_calls(raw_doc) == 0 and _raw_warnings(raw_doc) == []


def test_every_call_site_has_a_row():
    """A warning added without a row in SITES fails here, and so does one raised
    around the bindings' `warn`, which would carry no category."""
    in_source = Counter()
    raw = []
    for name, text in _rust_sources():
        in_source[name] = _warn_calls(text)
        raw += [f"{name}: {found}" for found in _raw_warnings(text)]
    for path in sorted((WORKSPACE / "python" / "hotcoco").rglob("*.py")):
        text = path.read_text()
        name = path.relative_to(WORKSPACE).as_posix()
        in_source[name] = _py_warn_calls(text)
        raw += [f"{name}: {found}" for found in _py_raw_warnings(text)]
    assert raw == [f"{LIB_RS}: PyErr::warn"], f"warnings raised outside `warn`: {raw}"
    in_source = +in_source  # drop files with no call
    covered = Counter(param.values[3] for param in SITES) + UNREACHABLE
    assert in_source == covered, f"warning calls per file {dict(in_source)}, rows per file {dict(covered)}"


def test_semantic_has_no_deviation_yet():
    ev = semantic.SemanticEval(2)
    ev.update(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
    assert ev.reference_deviations() == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ev.summarize()


LEAVES = (ReferenceDeviationWarning, CallOrderWarning, ConversionWarning, DatasetWarning)


def test_the_categories_nest_under_user_warning():
    assert issubclass(HotcocoWarning, UserWarning)
    for category in LEAVES:
        assert issubclass(category, HotcocoWarning)
        for other in LEAVES:
            assert category is other or not issubclass(category, other), (category, other)
    for category in (HotcocoWarning, *LEAVES):
        assert category.__name__ in hotcoco.__all__
        # The path a traceback prints is the one users import from.
        assert f"{category.__module__}.{category.__qualname__}" == f"hotcoco.{category.__name__}"


def test_the_base_is_never_emitted():
    """Every row and every ``warn`` call names a leaf, and every leaf hotcoco exports
    has a row.

    A site emitting ``HotcocoWarning`` itself could be silenced only by silencing
    every hotcoco warning; ``COCO.from_cvat`` did until ``ConversionWarning``.
    """
    exported = {
        getattr(hotcoco, name)
        for name in hotcoco.__all__
        if isinstance(getattr(hotcoco, name), type) and issubclass(getattr(hotcoco, name), HotcocoWarning)
    }
    assert exported == {HotcocoWarning, *LEAVES}
    assert {param.values[1] for param in SITES} == set(LEAVES)
    named = [f"{name}: {found}" for name, text in _rust_sources() for found in _non_leaf_warnings(text)]
    assert named == [], f"`warn` calls that name no leaf category: {named}"


def test_every_deviation_is_one_reference_deviation_warning():
    """``summarize()`` emits each entry of ``reference_deviations()`` under the category."""
    ev = _accumulated(iouThrs=[0.5], recThrs=[i / 10 for i in range(11)], useCats=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev.summarize()
    expected = [f"hotcoco: {d}" for d in ev.reference_deviations()]
    assert len(expected) >= 3
    assert [str(w.message) for w in caught] == expected
    assert {w.category for w in caught} == {ReferenceDeviationWarning}


def test_ignoring_deviations_leaves_call_order_warnings_on():
    """The filter the docs show: silence a deliberate configuration, keep the rest."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=ReferenceDeviationWarning)
        _accumulated(maxDets=[1, 10, 500]).summarize()
        with pytest.raises(CallOrderWarning):
            _ev().accumulate()


def test_a_text_filter_written_for_user_warning_still_matches():
    """RF-DETR silences the max_dets warning by its text, under ``UserWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="^hotcoco: max_dets differ from expected", category=UserWarning)
        _accumulated(maxDets=[1, 10, 500]).summarize()
