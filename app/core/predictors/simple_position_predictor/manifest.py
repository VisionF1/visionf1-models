from __future__ import annotations
import os, json
from typing import Any, Dict, Optional
from .constants import MANIFEST_PATHS

class InferenceManifest:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.manifest: Optional[Dict[str, Any]] = None

    def log(self, msg: str):
        if not self.quiet:
            print(msg)

    def load(self) -> Optional[Dict[str, Any]]:
        for path in MANIFEST_PATHS:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                    return self.manifest
            except Exception:
                continue
        return None

    def feature_names(self) -> Optional[list[str]]:
        if not self.manifest:
            return None
        return self.manifest.get("feature_names")

    def manifest_dir(self) -> str:
        for path in MANIFEST_PATHS:
            if os.path.exists(path):
                return os.path.dirname(path)
        return os.getcwd()

    def get_encoder_spec(self, key: str):
        if not self.manifest:
            return None
        enc = self.manifest.get("encoders", {})
        return enc.get(key)

