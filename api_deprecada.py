'''
@app.post("/train-and-predict-all")
def train_and_predict_all():
    """
    Equivalente a la opción 6 del CLI:
      6) Entrenar (quali + carrera), predecir quali y luego carrera usando grilla
    Devuelve los paths de artifacts y el top 10 de carrera.
    """
    info = get_next_race_info()
    artifacts = pipeline.train_and_predict_all()

    top10 = []
    error_top10 = None
    try:
        top10 = build_race_top10(artifacts)
    except Exception as e:
        error_top10 = str(e)

    return {
        "status": "ok",
        "detail": "Entrenamiento y predicción de quali + carrera ejecutados",
        "next_race": info,
        "artifacts": artifacts,
        "race_top10": top10,
        "race_top10_error": error_top10,
    }
'''



'''
@app.post("/train/quali")
def train_quali():
    """
    Equivalente a la opción 4 del CLI:
      4) Entrenar modelo de quali basado en últimas qualis
    """
    info = get_next_race_info()
    ok = pipeline.train_quali_from_fp3(year=info["season"])
    return {
        "status": "ok" if ok else "error",
        "success": ok,
        "detail": "Entrenamiento de modelo de quali ejecutado",
        "next_race": info,
    }
'''


'''
@app.post("/collect-data")
def collect_data():
    """
    Equivalente a la opción 1 del CLI:
      1) Descargar datos + preprocesar
    """
    pipeline.collect_data()
    pipeline.preprocess_data()
    return {
        "status": "ok",
        "detail": "Datos descargados y procesados exitosamente",
    }


@app.post("/train-models")
def train_models():
    """
    Equivalente a la opción 2 del CLI:
      2) Entrenar modelos (con feature engineering avanzado)
    """
    pipeline.run()
    return {
        "status": "ok",
        "detail": "Modelos entrenados correctamente",
    }

'''


def build_race_top10(artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Versión API de lo que antes imprimías en consola para la opción 6.
    Lee el CSV y devuelve el top 10 como JSON.
    """
    import pandas as pd

    rp = artifacts.get("race_predictions", "app/models_cache/race_predictions_latest.csv")
    df = pd.read_csv(rp)
    df = df.sort_values("final_position").head(10)

    results = []
    for _, r in df.iterrows():
        results.append(
            {
                "final_position": int(r["final_position"]),
                "driver": r["driver"],
                "team": str(r["team"]),
                "score": float(
                    r.get("predicted_position", r.get("model_position_score", 0.0))
                ),
            }
        )
    return results
