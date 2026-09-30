"""CLI surface checks that need no data: flags every user tries first."""

from __future__ import annotations

import importlib.metadata
import sys

import pytest
from hotcoco import cli


def test_version_flag_prints_installed_version(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["coco", "--version"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"coco {importlib.metadata.version('hotcoco')}"
