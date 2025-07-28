# Configuración del rango de carreras para el entrenamiento

# Define el rango de carreras a utilizar para el entrenamiento
RACE_RANGE = {
    "start_year": 2024,
    "num_races": 24  # Reducido para pruebas iniciales
}

# Define otras configuraciones necesarias para el proyecto
FEATURES_TO_EXTRACT = [
    "driver",
    "best_lap_time",
    "sector_times",
    "clean_air_race_pace"
]

# Define el modelo a utilizar para el entrenamiento
MODELS = {
    "random_forest": "RandomForestPredictor",
    "xgboost": "XGBoostPredictor",
    "gradient_boosting": "GradientBoostingPredictor"
}