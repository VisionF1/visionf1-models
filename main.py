import sys
from app.core.pipeline import Pipeline
from app.config import RACE_RANGE, PREDICTION_CONFIG

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [1|2|3|4|5]")
        print("  1) Descargar datos")
        print("  2) Entrenar modelos (con feature engineering avanzado)")
        print("  3) Predicciones de posiciones para próxima carrera")
        print("  4) Entrenar modelo de quali basado en últimas qualis")
        print("  5) Predecir quali próxima carrera (sin FP3, según config)")
        sys.exit(1)
    print("")
    print(f"📅 Próxima carrera configurada: {PREDICTION_CONFIG['next_race']['race_name']}")
    print(f"🏁 Carrera #{PREDICTION_CONFIG['next_race'].get('race_number', 1)} de la temporada 2025")
    weather_emoji = PREDICTION_CONFIG.get("active_scenario_emoji", "")
    print(f"Condiciones meteorológicas para predicción: {PREDICTION_CONFIG['active_scenario']} {weather_emoji}")


    action = sys.argv[1]
    # Pipeline básico
    pipeline = Pipeline(RACE_RANGE)

    if action == "1":
        print("Descargando datos de FastF1...")
        pipeline.collect_data()
        pipeline.preprocess_data()
        print("Datos descargados y procesados exitosamente.")
    elif action == "2":
        print("🚀 Entrenando modelos...")
        pipeline.run()
    elif action == "3":
        pipeline.predict_next_race_positions()
    elif action == "4":
        print("🧪 Entrenando modelo de quali basado en últimas qualis...")
        ok = pipeline.train_quali_from_fp3(year=2025)
        if not ok:
            print("❌ Falló el entrenamiento del modelo de qualis recientes")
    elif action == "5":
        print("🎯 Prediciendo quali para próxima carrera (sin FP3)...")
        pipeline.predict_quali_next_race()
    else:
        print("Acción no válida. Usa '1', '2', '3', '4' o '5'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
