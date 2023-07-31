"""
Unit and regression test for the espfit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import espfit


def test_espfit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "espfit" in sys.modules
