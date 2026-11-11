"""Every ``COCOeval`` binding method that reads ``params`` goes through ``with_params``.

``ev.params`` is a Python object. The binding copies it into the core evaluator only
inside ``with_params``, so a method that reads ``self.inner`` directly sees the params
as they stood at the last synced call. Since the run-state gate compares the params an
accumulation recorded with the evaluator's own, such a method reports a retired run as
current: ``print_results()`` did, until review caught it.

The check is an allowlist, as ``crates/hotcoco/tests/architecture.rs``'s are: a direct
``self.inner`` read is allowed only for an accessor listed below with the reason it
never reads the live ``params``, and only the sync points touch ``self.inner`` whole.
"""

from __future__ import annotations

import re

from helpers import WORKSPACE, rust_code

LIB_RS = WORKSPACE / "crates" / "hotcoco-pyo3" / "src" / "lib.rs"

# Core accessors a method may call on `self.inner` without syncing, and why each
# never reads the live `params`.
PARAMS_FREE = {
    "accumulated": "the last accumulate()'s arrays, which record the params they were laid out under",
    "stats": "the last summary, kept as pycocotools keeps E.stats",
    "evaluated": "whether evaluate() has run",
    "eval_imgs": "rebuilt from the inputs evaluate() snapshotted",
    "iou_matrices": "read from the params evaluate() resolved",
    "coco_gt": "a dataset, which params do not change",
    "coco_dt": "a dataset, which params do not change",
    "hierarchy": "the Open Images hierarchy, a field beside params",
}

# The methods that move params between `ev.params` and the core evaluator.
SYNC_POINTS = {
    "with_params": "copies ev.params in, runs, and copies the result back",
    "set_params": "replaces both sides at once",
}


def _block(text, open_brace):
    """The text from the `{` at `open_brace` to its matching `}`."""
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : i + 1]
    raise ValueError("unbalanced braces")


def methods(source):
    """Each `fn` in the `impl PyCOCOeval` blocks, as (name, body)."""
    code = rust_code(source)
    for impl in re.finditer(r"\bimpl\s+PyCOCOeval\s*\{", code):
        block = _block(code, impl.end() - 1)
        pos = 1
        while (fn := re.compile(r"\bfn\s+(\w+)").search(block, pos)) is not None:
            body = _block(block, block.index("{", fn.end()))
            yield fn.group(1), body
            pos = block.index(body, fn.end()) + len(body)


def unsynced_reads(source):
    """(method, accessor) for each direct `self.inner` use outside the allowlist;
    `<whole>` for `self.inner` used without an accessor, which can reach anything."""
    found = []
    for name, body in methods(source):
        if name in SYNC_POINTS:
            continue
        for use in re.finditer(r"self\s*\.\s*inner\b(?:\s*\.\s*(\w+))?", body):
            accessor = use.group(1) or "<whole>"
            if accessor not in PARAMS_FREE:
                found.append((name, accessor))
    return found


def test_every_direct_read_of_the_evaluator_is_params_free():
    source = LIB_RS.read_text()
    found = {name for name, _ in methods(source)}
    # A parse that missed the methods would pass for nothing.
    assert {"evaluate", "accumulate", "print_results", "get_results", "get_eval", "with_params"} <= found
    assert unsynced_reads(source) == []


def test_the_check_catches_an_unsynced_read():
    unsynced = """
    impl PyCOCOeval {
        fn print_results(&self, py: Python<'_>) -> PyResult<()> {
            // self.inner.ignored_in_a_comment()
            let lines = self.inner.print_results_lines();
            print_lines(py, &lines)
        }
        fn aliased(&self) -> bool {
            let inner = &self.inner;
            inner.evaluated()
        }
        fn fine(&self) -> bool {
            self.inner.evaluated()
        }
    }
    """
    assert unsynced_reads(unsynced) == [("print_results", "print_results_lines"), ("aliased", "<whole>")]
