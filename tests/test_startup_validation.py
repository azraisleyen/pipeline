from pathlib import Path

import pytest

from src.common.exceptions import ModelPathError
from src.pipeline.startup_validation import StartupValidator


def base_config(tmp_path):
    return {
        "repo_root": str(tmp_path),
        "pipeline": {},
        "model_paths": {"task1": {}, "task3": {}},
        "task1": {"allow_missing_models": True},
        "task2": {},
        "task3": {"allow_unavailable": True},
        "online": {},
    }


def test_startup_validation_allows_degraded_missing_assets(tmp_path):
    assert StartupValidator(base_config(tmp_path)).validate() is True


def test_startup_validation_rejects_task3_archive(tmp_path):
    cfg = base_config(tmp_path)
    cfg["task3"] = {"allow_unavailable": False, "detector_model_path": "model.zip", "config_path": "cfg.yaml"}
    (tmp_path / "cfg.yaml").write_text("x: 1")
    with pytest.raises(ModelPathError):
        StartupValidator(cfg).validate()
