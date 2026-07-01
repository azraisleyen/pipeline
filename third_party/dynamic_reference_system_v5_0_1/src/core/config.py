
from pathlib import Path
import yaml


class Config:
    def __init__(self, config_path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config dosyası bulunamadı: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)

        if self.raw is None:
            self.raw = {}

        if not isinstance(self.raw, dict):
            raise ValueError("Config dosyası YAML dictionary formatında olmalı.")

        for key, value in self.raw.items():
            setattr(self, key, value)

        default_sections = [
            "paths",
            "runtime",
            "detector",
            "proposal",
            "grid_proposal",
            "contour_proposal",
            "sam_proposal",
            "embedding",
            "matching",
            "geometry",
            "mellin",
            "fusion",
            "tracking",
            "output",
            "evaluation",
        ]

        for section in default_sections:
            if not hasattr(self, section):
                setattr(self, section, {})

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def resolve_path(self, base_dir, path_value):
        if path_value is None:
            return None

        path = Path(path_value)

        if path.is_absolute():
            return path

        return Path(base_dir) / path

    def to_dict(self):
        return dict(self.raw)
