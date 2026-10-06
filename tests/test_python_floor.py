"""The oldest supported Python is written in seven places, and they must agree.

The places this test reads:

- ``requires-python`` in ``pyproject.toml``, which every other place is compared to
- the lowest ``Programming Language :: Python :: 3.X`` classifier in ``pyproject.toml``
- ``[tool.ruff] target-version`` in ``pyproject.toml``
- ``[tool.pyright] pythonVersion`` in ``pyproject.toml``
- the ``abi3-py3X`` feature of pyo3 in ``crates/hotcoco-pyo3/Cargo.toml``
- the lowest entry of the smoke-test ``python-version`` matrix in ``.github/workflows/ci.yml``
- the ``- Python 3.X+`` prerequisite line in ``CONTRIBUTING.md``

Raising the floor means editing every one of them. ``docs/benchmarks.md`` also names
hotcoco's Python versions, but it describes the release on PyPI, so it changes when the
release ships, not with this tree.
"""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+; the newest CI leg runs this test")

ROOT = Path(__file__).resolve().parent.parent


def _minor(version):
    return int(version.split(".")[1])


def _ci_matrix_floor():
    """The lowest 3.X in the smoke-test job's matrix, the workflow's only `python-version: [...]` list."""
    path = ".github/workflows/ci.yml"
    matrices = re.findall(r"python-version:\s*\[([^\]]*)\]", (ROOT / path).read_text())
    assert len(matrices) == 1, (
        f"could not find the smoke-test matrix in {path}: expected one flow-style "
        f"`python-version: [...]` list, found {len(matrices)}"
    )
    versions = re.findall(r"3\.\d+", matrices[0])
    assert versions, f"{path}: no 3.X version in the smoke-test matrix {matrices[0]!r}"
    return min(versions, key=_minor)


def _classifier_floor(classifiers):
    """The lowest `Programming Language :: Python :: 3.X` classifier."""
    versions = [c for c in classifiers if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)]
    assert versions, "pyproject.toml: no `Programming Language :: Python :: 3.X` classifier"
    return min(versions, key=lambda c: _minor(c.rsplit(" ", 1)[1]))


def _contributing_floor():
    """The `- Python 3.X+` line in CONTRIBUTING.md's prerequisites."""
    lines = re.findall(r"^- Python 3\.\d+\+$", (ROOT / "CONTRIBUTING.md").read_text(), re.MULTILINE)
    assert len(lines) == 1, f"CONTRIBUTING.md: expected one `- Python 3.X+` line, found {len(lines)}"
    return lines[0]


def test_python_floor_agrees_everywhere():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    cargo = tomllib.loads((ROOT / "crates/hotcoco-pyo3/Cargo.toml").read_text())
    (abi3,) = [f for f in cargo["dependencies"]["pyo3"]["features"] if f.startswith("abi3-")]

    # Each place: its spelling of the floor, and a pattern capturing the X of 3.X.
    sources = {
        "pyproject.toml requires-python": (pyproject["project"]["requires-python"], r">=\s*3\.(\d+)"),
        "pyproject.toml lowest Python classifier": (
            _classifier_floor(pyproject["project"]["classifiers"]),
            r"Programming Language :: Python :: 3\.(\d+)",
        ),
        "pyproject.toml [tool.ruff] target-version": (pyproject["tool"]["ruff"]["target-version"], r"py3(\d+)"),
        "pyproject.toml [tool.pyright] pythonVersion": (pyproject["tool"]["pyright"]["pythonVersion"], r"3\.(\d+)"),
        "crates/hotcoco-pyo3/Cargo.toml pyo3 feature": (abi3, r"abi3-py3(\d+)"),
        ".github/workflows/ci.yml smoke-test matrix, lowest entry": (_ci_matrix_floor(), r"3\.(\d+)"),
        "CONTRIBUTING.md prerequisites": (_contributing_floor(), r"- Python 3\.(\d+)\+"),
    }
    minors = {}
    for where, (spelling, pattern) in sources.items():
        m = re.fullmatch(pattern, spelling)
        assert m, f"{where}: cannot read a 3.X version from {spelling!r}"
        minors[where] = int(m.group(1))

    floor = minors.pop("pyproject.toml requires-python")
    wrong = [f"{where} says 3.{minor}" for where, minor in minors.items() if minor != floor]
    assert not wrong, f"pyproject.toml requires-python says 3.{floor}, but " + "; ".join(wrong)
