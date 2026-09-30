"""Type stubs for the detection family namespace.

Every name here is re-exported from the top level, so the signatures live in
``__init__.pyi`` and this file only restates the identity of each one. Keep the
two in sync — `tests/test_stubs.py` checks the names, not the signatures.
"""

from . import COCOeval as COCOeval
from . import Hierarchy as Hierarchy
from . import Params as Params
from . import compare as compare

__all__ = ["COCOeval", "Hierarchy", "Params", "compare"]
