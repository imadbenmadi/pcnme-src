import yaml

import pcnme
from pcnme.experiments.cli import main


def test_package_imports():
    assert pcnme.__version__ == "0.1.0"


def test_default_config_parses(default_config_path):
    cfg = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    assert {"task_types", "hardware", "scenarios", "dqfd", "nsga2"} <= cfg.keys()


def test_cli_reports_unimplemented_commands():
    assert main(["testbed"]) == 2
