"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before each test to ensure isolation."""
    from src.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
