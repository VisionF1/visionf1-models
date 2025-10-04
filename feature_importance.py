# analyze_feature_importance.py
import os
import numpy as np
import pandas as pd

from app.core.predictors.simple_position_predictor.predictor import SimplePositionPredictor

def main():
    predictor = SimplePositionPredictor(quiet=False)
    # Reutiliza el método agregado dentro de la clase:
    fi = predictor.explain_feature_importance(
        top_k=25,
        csv_path="app/models_cache/feature_importances.csv",
        png_path="app/models_cache/feature_importances.png",
        n_repeats=15
    )
    print("\n=== TOP 20 FEATURES ===")
    print(fi.head(20))

if __name__ == "__main__":
    main()
