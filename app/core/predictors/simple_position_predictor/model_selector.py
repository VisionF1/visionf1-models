from __future__ import annotations
import os, pickle
from typing import Any, Dict, Optional, Tuple
from .constants import MODEL_FILES, TRAINING_RESULTS_PKL, FEATURE_NAMES_PKL

class ModelSelector:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def log(self, msg: str):
        if not self.quiet:
            print(msg)

    def _load_trained_feature_names(self) -> Optional[list[str]]:
        for cand in [FEATURE_NAMES_PKL, "/mnt/data/feature_names.pkl"]:
            try:
                with open(cand, "rb") as f:
                    names = pickle.load(f)
                if isinstance(names, (list, tuple)):
                    return list(names)
            except Exception:
                continue
        return None

    def _load_training_metrics(self) -> Optional[Dict[str, Any]]:
        for cand in [TRAINING_RESULTS_PKL, "/mnt/data/training_results.pkl"]:
            try:
                with open(cand, "rb") as f:
                    return pickle.load(f)
            except Exception:
                continue
        return None

    def _select_best_model_from_metrics(self, metrics: Dict[str, Any]) -> Optional[str]:
        best = None; best_cv = float('inf')
        for name, m in metrics.items():
            if "error" in m:
                continue
            over = m.get("overfitting_score", 2.0)
            cv   = m.get("cv_mse_mean", float('inf'))
            if over < 1.3 and cv < best_cv:
                best = name; best_cv = cv
        return best

    def load_best_model(self) -> Tuple[Optional[str], Any, Optional[list[str]]]:
        metrics = self._load_training_metrics()
        trained_feats = self._load_trained_feature_names()
        pick = None
        if metrics:
            pick = self._select_best_model_from_metrics(metrics)
        order = [pick] if pick else ["XGBoost", "GradientBoosting", "RandomForest"]
        for name in [x for x in order if x]:
            path = MODEL_FILES.get(name)
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                self.log(f"✅ Modelo cargado: {name}")
                return name, model, trained_feats
            except Exception as e:
                self.log(f"❌ No pude cargar {name}: {e}")
        return None, None, trained_feats
