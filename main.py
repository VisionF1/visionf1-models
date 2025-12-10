from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import pandas as pd


from app.core.pipeline import Pipeline
from app.config import RACE_RANGE, PREDICTION_CONFIG, SCENARIO_EMOJIS


VALID_RACE_NAMES = [
    "Abu Dhabi Grand Prix",
    "Australian Grand Prix",
    "Austrian Grand Prix",
    "Azerbaijan Grand Prix",
    "Bahrain Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Canadian Grand Prix",
    "Chinese Grand Prix",
    "Dutch Grand Prix",
    "Emilia Romagna Grand Prix",
    "French Grand Prix",
    "Hungarian Grand Prix",
    "Italian Grand Prix",
    "Japanese Grand Prix",
    "Las Vegas Grand Prix",
    "Mexico City Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Qatar Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Spanish Grand Prix",
    "São Paulo Grand Prix",
    "United States Grand Prix",
]


VALID_WEATHER_SCENARIOS = [
    "dry",
    "hot",
    "wet",
    "storm",
    "cold",
]

app = FastAPI(
    title="F1 Prediction API",
    description="API para entrenar modelos y generar predicciones de F1",
    version="1.0.0",
)

# Instanciamos una sola vez el pipeline al arrancar la app
pipeline = Pipeline(RACE_RANGE)


class PredictParams(BaseModel):
    race_name: Optional[str] = None
    weather_scenario: Optional[str] = None


def get_next_race_info(
    race_name: str | None = None,
    weather_scenario: str | None = None,
) -> Dict[str, Any]:

    cfg = PREDICTION_CONFIG
    race_cfg = cfg["next_race"]

    final_race_name = race_name or race_cfg["race_name"]
    final_scenario = weather_scenario or cfg["active_scenario"]
    final_emoji = SCENARIO_EMOJIS.get(final_scenario, '')
    return {
        "race_name": final_race_name,
        "season": 2025,
        "active_scenario": final_scenario,
        "active_scenario_emoji": final_emoji,
    }

@app.get("/")
def root():
    """
    Endpoint básico de salud.
    Equivalente a un "ping" para ver si la API está arriba.
    """
    return {
        "status": "ok",
        "message": "API de modelo F1 funcionando",
    }


@app.get("/config-next-race")
def next_race():
    """
    Información de la carrera que se simulara por defecto
    """
    return get_next_race_info()


@app.post("/predict-race")
def predict_race(params: PredictParams):
    """
    Predicciones de posiciones para la próxima carrera.

    Body opcional:
    {
      "race_name": "São Paulo Grand Prix",
      "weather_scenario": "dry"
    }
    """
    cfg = PREDICTION_CONFIG

    # Valores actuales de config
    current_next_race = cfg["next_race"]
    current_scenario = cfg["active_scenario"]

    # Tomar del body o del config
    race_name = params.race_name or current_next_race["race_name"]
    scenario = params.weather_scenario or current_scenario

    # ---- VALIDACIONES ----

    # Validar carrera
    if params.race_name and race_name not in VALID_RACE_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_race_name",
                "message": "race_name debe estar entre las opciones permitidas",
                "valid_race_names": VALID_RACE_NAMES,
            },
        )

    # Validar escenario meteorológico
    if params.weather_scenario and scenario not in VALID_WEATHER_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_weather_scenario",
                "message": "weather_scenario inválido",
                "valid_weather_scenarios": VALID_WEATHER_SCENARIOS,
            },
        )

    # ---- GUARDAR CONFIG ORIGINAL ----
    original_next_race = current_next_race.copy()
    original_active_scenario = cfg["active_scenario"]
    original_active_emoji = cfg.get("active_scenario_emoji")

    try:
        # ---- OVERRIDE TEMPORAL ----
        cfg["next_race"]["race_name"] = race_name
        cfg["active_scenario"] = scenario

        # Crear info con los valores actualizados
        info = get_next_race_info(
            race_name=race_name,
            weather_scenario=scenario,
        )

        # ---- EJECUTAR PREDICCIÓN ----
        pipeline.predict_next_race_positions()
        results = build_race_full()

        return {
            "status": "ok",
            "detail": "Predicción de posiciones de carrera generada",
            "next_race": info,
            "race_predictions": results,

        }

    finally:
        # ---- RESTAURAR CONFIG GLOBAL ----
        cfg["next_race"] = original_next_race
        cfg["active_scenario"] = original_active_scenario
        cfg["active_scenario_emoji"] = original_active_emoji

@app.post("/predict-quali")
def predict_quali(params: PredictParams):
    """
    Predecir quali de la próxima carrera:

    Body opcional:
    {
      "race_name": "São Paulo Grand Prix",
      "weather_scenario": "wet"
    }
    """
    cfg = PREDICTION_CONFIG

    # Valores actuales de config
    current_next_race = cfg["next_race"]
    current_scenario = cfg["active_scenario"]

    # Tomar del body o del config
    race_name = params.race_name or current_next_race["race_name"]
    scenario = params.weather_scenario or current_scenario

    # ---- VALIDACIONES ----

    # Validar carrera
    if params.race_name and race_name not in VALID_RACE_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_race_name",
                "message": "race_name debe estar entre las opciones permitidas",
                "valid_race_names": VALID_RACE_NAMES,
            },
        )

    # Validar escenario meteorológico
    if params.weather_scenario and scenario not in VALID_WEATHER_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_weather_scenario",
                "message": "weather_scenario inválido",
                "valid_weather_scenarios": VALID_WEATHER_SCENARIOS,
            },
        )

    # ---- GUARDAR CONFIG ORIGINAL ----
    original_next_race = current_next_race.copy()
    original_active_scenario = cfg["active_scenario"]
    original_active_emoji = cfg.get("active_scenario_emoji")

    try:
        # ---- OVERRIDE TEMPORAL ----
        cfg["next_race"]["race_name"] = race_name
        cfg["active_scenario"] = scenario

        # Crear info con los valores actualizados
        info = get_next_race_info(
            race_name=race_name,
            weather_scenario=scenario,
        )

        # ---- EJECUTAR PREDICCIÓN ----
        pipeline.predict_quali_next_race()

        quali_top = build_quali_top(top=20)

        return {
            "status": "ok",
            "detail": "Predicción de quali generada",
            "next_race": info,
            "quali_predicts": quali_top,
        }

    finally:
        # ---- RESTAURAR CONFIG GLOBAL ----
        cfg["next_race"] = original_next_race
        cfg["active_scenario"] = original_active_scenario
        cfg["active_scenario_emoji"] = original_active_emoji


def build_quali_top(top=10) -> List[Dict[str, Any]]:
    """
    Lee el CSV de predicciones de quali y devuelve el top n en JSON.
    Usa las columnas:
    - driver, team, pred_rank, pred_best_quali_lap_s, pred_best_quali_lap
    """
    qp = "app/models_cache/quali_predictions_latest.csv"
    
    df = pd.read_csv(qp)

    # ordenar por ranking de quali y tomar top
    df = df.sort_values("pred_rank").head(top)

    results: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        item: Dict[str, Any] = {
            "driver": r["driver"],
            "team": r["team"],
            "race_name": r.get("race_name"),
            "pred_rank": int(r["pred_rank"]),
            "pred_best_quali_lap": r["pred_best_quali_lap"],
        }
        results.append(item)

    return results



def build_race_full() -> List[Dict[str, Any]]:
    """
    Lee el CSV de predicciones de carrera y devuelve TODAS las filas en JSON.
    Usa:
    - final_position, driver, team, model_position_score,
        grid_position, predicted_position
    """
    rp = "app/models_cache/race_predictions_latest.csv"
    df = pd.read_csv(rp)

    # ordenar por posición final
    df = df.sort_values("final_position")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        item: Dict[str, Any] = {
            "driver": r["driver"],
            "team": r["team"],
            "final_position": int(r["final_position"]),
        }
        rows.append(item)

    return rows

@app.post("/predict-all")
def predict_all(params: PredictParams):
    """
      6) Predecir quali y luego carrera usando lo predicho anteriormente.

    Permite opcionalmente indicar:
      - race_name: nombre de la carrera (debe estar en config)
      - weather_scenario: escenario meteorológico (dry, hot, wet, storm, cold)

    Devuelve:
      - top 10 de la quali
      - predicción completa de carrera (todas las posiciones)
    """
    cfg = PREDICTION_CONFIG

    current_next_race = cfg["next_race"]
    current_scenario = cfg["active_scenario"]

    # Valores pedidos por la request (o defaults del config)
    race_name = params.race_name or current_next_race["race_name"]
    scenario = params.weather_scenario or current_scenario

    # Validar escenario meteorológico contra config
    valid_scenarios = VALID_WEATHER_SCENARIOS
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_weather_scenario",
                "message": "weather_scenario debe ser uno de los escenarios definidos en config",
                "valid_weather_scenarios": sorted(valid_scenarios),
            },
        )

    allowed_races = VALID_RACE_NAMES
    if params.race_name and race_name not in allowed_races:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_race_name",
                "message": "race_name debe estar entre las opciones definidas en config",
                "valid_race_names": allowed_races,
            },
        )

    # Guardar original para restaurar luego
    original_next_race = current_next_race.copy()
    original_active_scenario = cfg["active_scenario"]
    original_active_emoji = cfg.get("active_scenario_emoji")

    try:
        # Override temporal de config para esta predicción
        cfg["next_race"]["race_name"] = race_name
        cfg["active_scenario"] = scenario

        info = get_next_race_info(
            race_name=race_name,
            weather_scenario=scenario,
        )        
        artifacts = pipeline.predict_all()

        race_top10: List[Dict[str, Any]] = []
        race_full: List[Dict[str, Any]] = []
        quali_top10: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}

        try:
            race_full = build_race_full()
        except Exception as e:
            errors["race_full"] = str(e)

        try:
            quali_top10 = build_quali_top(top=10)
        except Exception as e:
            errors["quali_top10"] = str(e)

        return {
            "status": "ok" if not errors else "partial_ok",
            "detail": "Predicción de quali + carrera ejecutadas",
            "next_race": info,
            "quali_top10": quali_top10,
            "race_predictions_full": race_full,
            "errors": errors or None,
        }
    finally:
        # Restaurar config global
        cfg["next_race"] = original_next_race
        cfg["active_scenario"] = original_active_scenario
        cfg["active_scenario_emoji"] = original_active_emoji
