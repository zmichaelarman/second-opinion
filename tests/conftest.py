import shutil
from pathlib import Path

import pytest

from second_opinion.config import load_settings

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "test-key-not-real"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Settings rooted in a temp dir (real config files, empty runs/) with a fake API key."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    defaults = tmp_path / "config" / "defaults.yaml"
    defaults.write_text(defaults.read_text().replace("\nrpm: 40 ", "\nrpm: 60000 "))  # subprocess/testkit reloads from disk
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_KEY)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    monkeypatch.delenv("SECOND_OPINION_BUDGET_USD", raising=False)
    s = load_settings(tmp_path)
    s.defaults["rpm"] = 60_000  # no pacing delays in tests
    return s
