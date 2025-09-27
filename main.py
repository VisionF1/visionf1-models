import sys
from app.core.pipeline import Pipeline
from app.config import RACE_RANGE, PREDICTION_CONFIG

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [1|2|3|4|5|6]")
        print("  1) Descargar datos")
        print("  2) Entrenar modelos (con feature engineering avanzado)")
        print("  3) Predicciones de posiciones para próxima carrera")
        print("  4) Entrenar modelo de quali basado en últimas qualis")
        print("  5) Predecir quali próxima carrera (sin FP3, según config)")
        print("  6) Entrenar (quali + carrera), predecir quali y luego carrera usando grilla")
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
    elif action == "6":
        print("🧩 Entrenando y prediciendo quali + carrera...")
        artifacts = pipeline.train_and_predict_all()
        print("Listo. Archivos:")
        for k, v in artifacts.items():
            print(f" - {k}: {v}")
        # Imprimir Top-10 de carrera
        try:
            import pandas as pd
            rp = artifacts.get("race_predictions", "app/models_cache/race_predictions_latest.csv")
            df = pd.read_csv(rp)
            df = df.sort_values("final_position").head(10)
            print("\n🏁 Top 10 Predicción Carrera")
            print(f"{'Pos':<4} {'Piloto':<6} {'Equipo':<16} {'Score':<8} {'Grid':<4}")
            for _, r in df.iterrows():
                print(f"P{int(r['final_position']):<3} {r['driver']:<6} {str(r['team'])[:16]:<16} {float(r.get('predicted_position', r.get('model_position_score', 0))):<8.3f} {int(r.get('grid_position', 0)):<4}")
        except Exception as e:
            print(f"⚠️ No se pudo imprimir top-10 de carrera: {e}")
    else:
        print("Acción no válida. Usa '1', '2', '3', '4' o '5'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
