"""Pytest configuration for reach_avoid tests.

Adds the VCIP vendor directory to sys.path so that `src.models.*` imports resolve.
"""

import sys
from pathlib import Path

VCIP_DIR = Path(__file__).resolve().parents[2] / "vendor" / "VCIP"
if str(VCIP_DIR) not in sys.path:
    sys.path.insert(0, str(VCIP_DIR))
