"""Verify that __init__.pyi covers every public symbol in the hotcoco module.

Run with: uv run pytest scripts/test_stubs.py -v
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import hotcoco


STUB_PATH = Path(__file__).resolve().parent.parent / "python" / "hotcoco" / "__init__.pyi"


def _parse_stub_names() -> dict[str, set[str]]:
    """Parse the .pyi file and return {class_name: {method_names}} and top-level names."""
    source = STUB_PATH.read_text()
    tree = ast.parse(source)

    top_level: set[str] = set()
    classes: dict[str, set[str]] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level.add(node.name)
        elif isinstance(node, ast.ClassDef):
            top_level.add(node.name)
            members: set[str] = set()
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
            classes[node.name] = members
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level.add(target.id)

    return {"__top__": top_level, **classes}


def _public_names(obj: object) -> set[str]:
    """Get public attribute names (no underscore prefix)."""
    return {name for name in dir(obj) if not name.startswith("_")}


def test_stub_file_exists():
    assert STUB_PATH.exists(), f"Missing stub file: {STUB_PATH}"


def test_py_typed_exists():
    py_typed = STUB_PATH.parent / "py.typed"
    assert py_typed.exists(), "Missing py.typed marker"


def test_top_level_exports_covered():
    """Every public name in hotcoco should appear in the stub."""
    stub_names = _parse_stub_names()["__top__"]
    runtime_names = _public_names(hotcoco)

    # These are re-exports or internal names we don't need to stub
    skip = {"LVIS", "LVISeval", "LVISResults", "CocoDetection", "CocoEvaluator", "hotcoco", "integrations"}
    runtime_names -= skip

    missing = runtime_names - stub_names
    assert not missing, f"Public names missing from stubs: {sorted(missing)}"


def test_coco_methods_covered():
    """Every public method on COCO should appear in the stub."""
    stub_members = _parse_stub_names().get("COCO", set())
    runtime_members = _public_names(hotcoco.COCO)

    missing = runtime_members - stub_members
    assert not missing, f"COCO methods missing from stubs: {sorted(missing)}"


def test_cocoeval_methods_covered():
    """Every public method on COCOeval should appear in the stub."""
    stub_members = _parse_stub_names().get("COCOeval", set())
    runtime_members = _public_names(hotcoco.COCOeval)

    missing = runtime_members - stub_members
    assert not missing, f"COCOeval methods missing from stubs: {sorted(missing)}"


def test_params_attrs_covered():
    """Every public attr on Params should appear in the stub."""
    stub_members = _parse_stub_names().get("Params", set())
    runtime_members = _public_names(hotcoco.Params)

    missing = runtime_members - stub_members
    assert not missing, f"Params attrs missing from stubs: {sorted(missing)}"


def test_mask_methods_covered():
    """Every public function in mask should appear in the stub."""
    stub_members = _parse_stub_names().get("mask", set())
    runtime_members = _public_names(hotcoco.mask)

    missing = runtime_members - stub_members
    assert not missing, f"mask methods missing from stubs: {sorted(missing)}"


def test_hierarchy_methods_covered():
    """Every public method on Hierarchy should appear in the stub."""
    stub_members = _parse_stub_names().get("Hierarchy", set())
    runtime_members = _public_names(hotcoco.Hierarchy)

    missing = runtime_members - stub_members
    assert not missing, f"Hierarchy methods missing from stubs: {sorted(missing)}"
