import sys
from app.core.pipeline import Pipeline
from app.config import RACE_RANGE

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [descargar|entrenar|predecir]")
        sys.exit(1)

    action = sys.argv[1]
    pipeline = Pipeline(RACE_RANGE)

    if action == "d":
        print("Descargando datos de FastF1...")
        pipeline.collect_data()
        pipeline.preprocess_data()
        print("Datos descargados y procesados exitosamente.")
    elif action == "e":
        print("Entrenando modelos...")
        pipeline.run()
    elif action == "p":
        pipeline.make_predictions()
    else:
        print("Acción no válida. Usa 'd', 'e' o 'p'.")
        sys.exit(1)

if __name__ == "__main__":
    main()