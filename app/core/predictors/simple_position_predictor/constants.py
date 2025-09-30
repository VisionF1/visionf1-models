MODEL_FILES = {
    "RandomForest": "app/models_cache/randomforest_model.pkl",
    "XGBoost": "app/models_cache/xgboost_model.pkl",
    "GradientBoosting": "app/models_cache/gradientboosting_model.pkl",
}

ENCODER_FILES = {
    "driver": "app/models_cache/driver_encoder.pkl",
    "team": "app/models_cache/team_encoder.pkl",
    "race_name": "app/models_cache/race_name_encoder.pkl",
    "circuit_type": "app/models_cache/circuit_type_encoder.pkl",
}

TRAINING_RESULTS_PKL = "app/models_cache/training_results.pkl"
FEATURE_NAMES_PKL    = "app/models_cache/feature_names.pkl"
MANIFEST_PATHS = [
    "app/models_cache/inference_manifest.json",
    "/mnt/data/inference_manifest.json",
]

# Opcionales usados en features
CACHED_DATA_PKL      = "app/models_cache/cached_data.pkl"
BEFORE_FE_PATH       = "app/models_cache/dataset_before_training_latest.csv"
AFTER_FE_PATH        = "app/models_cache/dataset_after_feature_engineering_latest.csv"

# Dataset de inferencia (debug)
INFERENCE_OUT        = "app/models_cache/inference_inference_dataset.csv"

