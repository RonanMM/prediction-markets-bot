"""
    Dummy conftest.py for raincheck.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pytest
import sys
from pathlib import Path

# Canonical import contract (see implementation_plan.md):
#   - `src/` on path           -> `raincheck` package (PyScaffold src-layout)
#   - `src/polymarket_weather/` -> the source root; all intra-project imports are bare
#     (`from config import ...`, `from engine import ...`, `from resolution_anchors import ...`)
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "polymarket_weather"))
