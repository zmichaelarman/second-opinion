"""Offline checks that the CLI loads and the config files are well-formed."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from second_opinion.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_lists_m0_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "models" in result.output
    assert "smoke" in result.output


def test_every_price_row_has_rates_source_and_as_of():
    prices = yaml.safe_load((ROOT / "config" / "prices.yaml").read_text())
    assert prices["models"]
    for model_id, row in prices["models"].items():
        assert row["input"] >= 0 and row["output"] >= 0, model_id
        assert row["source"], model_id
        assert isinstance(row["as_of"], str), f"{model_id}: quote as_of so it stays JSON-serializable"


def test_defaults_have_required_keys():
    defaults = yaml.safe_load((ROOT / "config" / "defaults.yaml").read_text())
    for key in ("rpm", "concurrency", "n_screen", "n_escalate", "alpha"):
        assert key in defaults, key
