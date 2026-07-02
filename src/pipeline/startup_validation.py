from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.exceptions import ConfigError, ModelPathError
from src.common.model_registry import ModelRegistry

REQUIRED_CONFIG_SECTIONS = ("pipeline", "model_paths", "task1", "task2", "task3", "online")
TASK1_MODEL_KEYS = ("task1.human", "task1.vehicle", "task1.uap_uai", "task1.landing_classifier")


class StartupValidator:
    """Fail-fast runtime preflight checks for production simulation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.repo_root = Path(config.get("repo_root", "."))

    def validate(self) -> bool:
        self._validate_sections()
        self._validate_task1_models()
        self._validate_task3_assets()
        self._validate_online_config_if_required()
        return True

    def _validate_sections(self) -> None:
        missing = [section for section in REQUIRED_CONFIG_SECTIONS if section not in self.config]
        if missing:
            raise ConfigError(f"Missing required config section(s): {', '.join(missing)}")

    def _validate_task1_models(self) -> None:
        task1 = self.config.get("task1", {})
        if not task1.get("enabled", True) or task1.get("allow_missing_models", False):
            return
        registry = ModelRegistry(self.config, self.repo_root)
        for key in TASK1_MODEL_KEYS:
            registry.resolve(key, required=True)

    def _validate_task3_assets(self) -> None:
        task3 = self.config.get("task3", {})
        if not task3.get("enabled", True) or task3.get("allow_unavailable", False):
            return

        model_path = task3.get("detector_model_path") or self.config.get("model_paths", {}).get("task3", {}).get("detector")
        if not model_path:
            raise ModelPathError("Task3 detector model path is not configured")
        model = Path(model_path)
        if not model.is_absolute():
            model = self.repo_root / model
        if model.suffix.lower() == ".zip":
            raise ModelPathError(f"Task3 detector model must be an extracted weight, not archive: {model}")
        if not model.exists():
            raise ModelPathError(f"Required Task3 detector model file is missing: {model}")

        cfg_path = Path(task3.get("config_path", ""))
        if not cfg_path.is_absolute():
            cfg_path = self.repo_root / cfg_path
        if not cfg_path.exists():
            raise ConfigError(f"Task3 v5.0.1 config file is missing: {cfg_path}")

        reference_dir = task3.get("reference_dir")
        if reference_dir:
            ref = Path(reference_dir)
            if not ref.is_absolute():
                ref = self.repo_root / ref
            if not ref.exists():
                raise ConfigError(f"Task3 reference directory is missing: {ref}")

    def _validate_online_config_if_required(self) -> None:
        online = self.config.get("online", {})
        if online.get("required", False) and not online.get("base_url"):
            raise ConfigError("online.base_url is required when online.required is true")
