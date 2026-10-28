"""Shared pytest fixtures. Fixtures are added here as milestones need them."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    return DEFAULT_CONFIG


@pytest.fixture
def seed() -> int:
    """A fixed training-range seed (config fast_mode.seed_split). Tests never use evaluation seeds."""
    return 2000
