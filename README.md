# F1 Predictor

Este proyecto es un predictor de tiempos de carrera de Fórmula 1 utilizando modelos de machine learning. Se basa en la API de FastF1 para recolectar datos de carreras y extraer características relevantes para el entrenamiento de modelos de predicción.

## Estructura del Proyecto

El proyecto tiene la siguiente estructura:

```
f1_predictor/
├── app/
│   ├── core/
│   │   ├── predictors/          # Modelos (sklearn, xgboost, gradient boosting)
│   │   ├── features/            # Extracción de features (driver, lap times, sector times, pace sin tráfico)
│   │   └── pipeline.py          # Orquesta el flujo
│   ├── data/
│   │   ├── collectors/          # Usa fastf1 API para bajar los datos
│   │   └── preprocessors/       # Limpia y prepara los datos
│   ├── models/                  # Modelos entrenados
│   └── config.py                # Permite definir el rango de carreras para entrenar
├── main.py                      # Punto de entrada: entrena o predice
└── requirements.txt             # Dependencias del proyecto
```

## Requisitos

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes dependencias:

- fastf1
- scikit-learn
- xgboost

Puedes instalar las dependencias utilizando el siguiente comando:

```
pip install -r requirements.txt
```

## Uso

Para entrenar o predecir, utiliza el archivo `main.py`. Puedes especificar el rango de carreras en `config.py`.

Ejemplo de uso:

```
python main.py --train
```

o

```
python main.py --predict
```

## Contribuciones

Las contribuciones son bienvenidas. Si deseas agregar nuevas características o mejorar el proyecto, siéntete libre de hacer un fork y enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.