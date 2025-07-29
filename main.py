import sys
from app.core.pipeline import Pipeline
from app.config import RACE_RANGE, PREDICTION_CONFIG

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [1|2|3]")
        print("  1) Descargar datos")
        print("  2) Entrenar modelos")
        print("  3) Predicciones de posiciones para próxima carrera")
        print("")
        print(f"📅 Próxima carrera configurada: {PREDICTION_CONFIG['next_race']['race_name']}")
        print(f"🏁 Carrera #{PREDICTION_CONFIG['next_race'].get('race_number', 1)} de la temporada 2025")
        sys.exit(1)

    action = sys.argv[1]
    pipeline = Pipeline(RACE_RANGE)

    if action == "1":
        print("Descargando datos de FastF1...")
        pipeline.collect_data()
        pipeline.preprocess_data()
        print("Datos descargados y procesados exitosamente.")
    elif action == "2":
        print("Entrenando modelos...")
        pipeline.run()
    elif action == "3":
        pipeline.predict_next_race_positions()
    else:
        print("Acción no válida. Usa '1', '2' o '3'.")
        sys.exit(1)

if __name__ == "__main__":
    main()