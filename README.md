# VisionF1 Models - Predictor de Posiciones F1 2025

Este proyecto es un sistema avanzado de predicción de posiciones de Fórmula 1 utilizando modelos de machine learning con cross-validation. Combina datos históricos de FastF1 con configuraciones específicas de la temporada 2025 para generar predicciones realistas de carreras.

## 🏆 Características Principales

- **Cross-Validation Robusto**: TimeSeriesSplit para datos temporales + detección automática de overfitting
- **Selección Inteligente de Modelos**: Elige automáticamente el mejor modelo basado en métricas de rendimiento
- **Predicciones Híbridas**: Combina ML histórico (25%) con configuración 2025 actualizada (75%)
- **Sistema de Adaptación Progresiva**: Penalizaciones que disminuyen para rookies y cambios de equipo
- **Cache Inteligente**: Evita descargas innecesarias de datos FastF1

## 📊 Modelos Soportados

- **RandomForest**: Más estable, resistente al overfitting
- **XGBoost**: Balanceado, con regularización L1/L2
- **GradientBoosting**: Alta precisión, controlado contra overfitting

Todos los modelos incluyen:
- Optimización automática de hiperparámetros con GridSearchCV
- Detección de overfitting con scoring personalizado
- Métricas completas: CV MSE, Test R², Overfitting Score

## 🗂️ Estructura del Proyecto

```
visionf1-models/
├── app/
│   ├── core/
│   │   ├── predictors/          # Modelos ML (RandomForest, XGBoost, GradientBoosting)
│   │   ├── training/            # Entrenamiento con CV y preparación de datos
│   │   │   ├── model_trainer.py      # Cross-validation + optimización
│   │   │   └── data_preparer.py      # Preparación robusta de características
│   │   ├── features/            # Extracción de características avanzadas
│   │   │   └── feature_extractor.py  # FP1/FP2/FP3, Quali, Race, Derivadas
│   │   ├── adapters/            # Sistema de adaptación progresiva
│   │   ├── utils/               # Utilidades (RangeBuilder, futuros DataUtils)
│   │   └── pipeline.py          # Orquesta todo el flujo
│   ├── data/
│   │   ├── collectors/          # Recolección inteligente con cache
│   │   │   └── fastf1_collector.py   # Datos completos: FP1/FP2/FP3/Q/R
│   │   └── preprocessors/       # Limpieza y preparación
│   ├── models_cache/            # Modelos entrenados + cache de datos
│   │   ├── training_results.pkl      # Métricas para selección automática
│   │   ├── model_comparison.txt      # Resumen legible de modelos
│   │   └── *.pkl                     # Modelos optimizados guardados
│   └── config.py                # Configuración completa 2025
├── main.py                      # Punto de entrada con 3 modos
└── requirements.txt             # Dependencias del proyecto
```

## 🚀 Instalación y Configuración

### 1. Dependencias

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `fastf1` - API oficial de datos F1
- `scikit-learn` - Modelos ML y métricas
- `xgboost` - Algoritmo de boosting avanzado
- `pandas`, `numpy` - Manipulación de datos
- `matplotlib` - Visualizaciones

### 2. Configuración

El archivo [`app/config.py`](app/config.py) contiene toda la configuración:

```python
# Rango de datos para entrenamiento
RACE_RANGE = {
    "years": [2024, 2025],
    "max_races_per_year": 24,
    "auto_detect_available": True
}

# Predicción de próxima carrera
PREDICTION_CONFIG = {
    "next_race": {
        "race_name": "Hungarian Grand Prix",
        "race_number": 13  # Carrera actual de la temporada
    }
}

# Pilotos y equipos 2025 actualizados
DRIVERS_2025 = {
    "NOR": {"team": "McLaren", "tier": 1, "expected_range": (1, 4)},
    "HAM": {"team": "Ferrari", "tier": 2, "team_change": True},
    # ... configuración completa de 20 pilotos
}
```

## 🎯 Uso del Sistema

### Modo 1: Descargar Datos
```bash
python main.py 1
```
- Descarga datos frescos de FastF1 (FP1, FP2, FP3, Clasificación, Carrera)
- Usa cache inteligente para evitar descargas repetidas
- Procesa y limpia los datos automáticamente

### Modo 2: Entrenar Modelos
```bash
python main.py 2
```
- Entrena 3 modelos con cross-validation robusto
- Optimiza hiperparámetros automáticamente con GridSearchCV
- Detecta y previene overfitting
- Guarda métricas detalladas para selección automática

**Salida esperada:**
```
🚂 Entrenando modelos con Cross-Validation...
📊 Usando TimeSeriesSplit con 5 splits

==================================================
🔍 ENTRENANDO RandomForest CON CROSS-VALIDATION
==================================================
🔧 Optimizando hiperparámetros para RandomForest...
✅ Mejores parámetros: {'max_depth': 5, 'n_estimators': 100}
✅ Mejor CV score: 16.5274

📊 RESULTADOS DETALLADOS - RandomForest
----------------------------------------
🔄 Cross-Validation MSE: 16.5274 ± 28.3092
🎯 Test MSE: 0.9731
📈 Test R²: 0.9765
✅ Overfitting Score: 0.16 (Bueno)

🏆 MEJOR MODELO: RandomForest
   📊 CV MSE: 16.5274
   🎯 Sin overfitting significativo
```

### Modo 3: Predicciones
```bash
python main.py 3
```
- Selecciona automáticamente el mejor modelo entrenado
- Genera predicciones híbridas (ML + configuración 2025)
- Aplica sistema de adaptación progresiva
- Guarda resultados en CSV

**Salida esperada:**
```
🔍 Analizando métricas de modelos...
   ✅ RandomForest: CV=16.5274, Overfitting=0.16, R²=0.9765
   🚫 GradientBoosting: Rechazado (Overfitting=138.56)

🏆 MEJOR MODELO SELECCIONADO: RandomForest

================================================================================
🏆 PREDICCIONES ML + CONFIGURACIÓN 2025 - CARRERA #Hungarian Grand Prix
================================================================================
Pos  Piloto Equipo           Tier  Tipo            Fuente       Conf.
--------------------------------------------------------------------------------
P1   NOR    McLaren          T1    👤 Veterano     🔥 ML+Config  70%
P2   PIA    McLaren          T1    👤 Veterano     🔥 ML+Config  70%
P3   VER    Red Bull Racing  T2    👤 Veterano     🔥 ML+Config  72%
...
```

## 🧠 Sistema de Machine Learning

### Cross-Validation Inteligente
- **TimeSeriesSplit**: Para datos temporales (carreras cronológicas)
- **KFold**: Backup para datasets pequeños
- **Detección de Overfitting**: Score = Test MSE / Train MSE
- **Selección Automática**: Mejor modelo por CV score sin overfitting

### Características Extraídas
El sistema extrae **50+ características** automáticamente:

**Práctica Libre (FP1/FP2/FP3):**
- Mejores tiempos, tiempos promedio, número de vueltas
- Tiempos por sector, consistencia, progresión

**Clasificación:**
- Posiciones Q1/Q2/Q3, tiempos por sesión
- Posición de parrilla, sectores de clasificación

**Carrera:**
- Posición final, puntos, mejor vuelta
- Ritmo en aire limpio, sectores de carrera
- Vueltas totales, status de finalización

**Características Derivadas:**
- Posiciones ganadas (quali → carrera)
- Cambios de parrilla → carrera
- Consistencia en práctica, progresión

### Prevención de Overfitting
```python
# Criterios automáticos
if overfitting_score < 1.1:
    status = "✅ Excelente"
elif overfitting_score < 2.0:  
    status = "⚠️ Aceptable"
else:
    status = "🚨 Rechazado"

# Solo se guardan modelos confiables
if metrics['overfitting_score'] < 10.0:
    self._save_model(name, optimized_model)
```

## 📈 Sistema de Predicción Avanzado

### Predicciones Híbridas
- **25% Modelo ML**: Basado en datos históricos 2024-2025
- **75% Configuración 2025**: Expectativas actualizadas por equipo/piloto

### Adaptación Progresiva
El sistema penaliza automáticamente rookies y cambios de equipo:

```python
# Rookies: penalización que disminuye en 8 carreras
"ANT": {"rookie": True}      # -2.5 → -0 posiciones progresivamente

# Cambios de equipo: adaptación en 5 carreras  
"HAM": {"team_change": True} # -1.5 → -0 posiciones progresivamente
```

### Selección Inteligente de Modelos
```python
# El sistema elige automáticamente basado en:
def _select_best_model_from_metrics(self, training_metrics):
    viable_models = []
    
    for model_name, metrics in training_metrics.items():
        cv_score = metrics['cv_mse_mean']
        overfitting_score = metrics['overfitting_score'] 
        test_r2 = metrics['test_r2']
        
        # Solo modelos sin overfitting severo
        if overfitting_score < 2.0 and test_r2 > 0.5:
            viable_models.append(model)
    
    # Mejor modelo = menor CV score + sin overfitting
    best_model = min(viable_models, key=lambda x: x['overall_score'])
```

## 📁 Archivos de Salida

- **`app/models_cache/realistic_predictions_2025.csv`**: Predicciones detalladas
- **`app/models_cache/model_comparison.txt`**: Comparación legible de modelos
- **`app/models_cache/training_results.pkl`**: Métricas completas para selección automática

## 🔧 Personalización Avanzada

### Agregar Nuevos Pilotos
```python
# En app/config.py
DRIVERS_2025["NEW"] = {
    "team": "Nuevo Equipo",
    "tier": 3,
    "expected_range": (10, 15),
    "rookie": True  # Para aplicar adaptación progresiva
}
```

### Modificar Algoritmos de ML
```python
# En app/core/training/model_trainer.py
def _initialize_models_with_hyperparams(self):
    return {
        'MiNuevoModelo': {
            'model_class': MiModeloPersonalizado,
            'param_grid': {'param1': [1, 2, 3]}
        }
    }
```

### Cambiar Estrategia de Predicción
```python
# En app/config.py
DATA_IMPORTANCE = {
    "ml_weight": 0.50,     # 50% modelo ML
    "config_weight": 0.50  # 50% configuración
}
```

## 📊 Métricas y Evaluación

### Métricas de Cross-Validation
- **CV MSE Mean/Std**: Error cuadrático medio con desviación
- **Test R²**: Coeficiente de determinación en datos de prueba
- **Overfitting Score**: Ratio Test MSE / Train MSE
- **Train/Test MAE**: Error absoluto medio

### Criterios de Selección
1. **Sin overfitting severo** (score < 2.0)
2. **Buen rendimiento en CV** (MSE bajo)
3. **Alta explicabilidad** (R² > 0.5)
4. **Estabilidad temporal** (TimeSeriesSplit consistente)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. **Fork** el repositorio
2. **Crea una rama** para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. **Commit** tus cambios (`git commit -m 'Agregar nueva característica'`)
4. **Push** a la rama (`git push origin feature/nueva-caracteristica`)
5. **Abre un Pull Request**

### Áreas de Mejora Sugeridas
- Implementar modelos de deep learning (LSTM, Transformers)
- Agregar datos meteorológicos y de neumáticos
- Crear interfaz web con visualizaciones interactivas
- Expandir a predicciones de temporada completa
- Implementar backtesting automático

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Consulta el archivo `LICENSE` para más detalles.

---

**Desarrollado con 🏎️ para la comunidad de Fórmula 1**

*Sistema de predicción avanzado que combina machine learning robusto con conocimiento específico del deporte para generar predicciones realistas y actualizadas de la temporada 2025.*