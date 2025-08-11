# Guía de Migración: Soporte Histórico 2022-2025

## 🔄 Cambios Implementados

### 1. **Sistema de Configuración Histórica**
- **Archivo**: `app/config_historical.py`
- **Funcionalidad**: 
  - Configuración de drivers por año (2022-2025)
  - Mapeos de equipos que cambiaron de nombre
  - Funciones para obtener drivers activos por año

### 2. **Race Range Builder Adaptativo**
- **Archivo**: `app/core/utils/race_range_builder.py`
- **Cambios**:
  - Eliminadas verificaciones hardcoded de año 2025
  - Uso de fecha actual dinámicamente
  - Soporte para calendarios históricos

### 3. **Feature Extractor Histórico**
- **Archivo**: `app/core/features/historical_feature_extractor.py`
- **Funcionalidades**:
  - Adaptación a características disponibles por año
  - Normalización entre años para comparabilidad
  - Estimación de métricas faltantes en años anteriores

### 4. **Collector Histórico**
- **Archivo**: `app/data/collectors/historical_fastf1_collector.py`
- **Características**:
  - API específica por año
  - Manejo de formatos de datos diferentes
  - Fallback para datos no disponibles

### 5. **Adaptador Progresivo Mejorado**
- **Archivo**: `app/core/adapters/progressive_adapter.py`
- **Cambios**:
  - Soporte para año objetivo configurable
  - Uso de configuración histórica de drivers

### 6. **Pipeline Inteligente**
- **Archivo**: `app/core/pipeline.py`
- **Mejoras**:
  - Detección automática de años históricos
  - Selección de collector apropiado

## 🚀 Cómo Usar el Sistema Mejorado

### Configurar Años de Entrenamiento
```python
# En app/config.py
RACE_RANGE = {
    "years": [2022, 2023, 2024, 2025],  # Ahora soporta desde 2022
    "max_races_per_year": 24,
    "include_current_year": True,
    "auto_detect_available": True,
    "stop_on_future_races": True
}
```

### Uso Normal
```bash
# 1. Descargar datos históricos (2022-2025)
python main.py 1

# 2. Entrenar con datos históricos
python main.py 2

# 3. Predicciones con contexto histórico
python main.py 3
```

## 📊 Características Históricas Soportadas

### Por Año
- **2022**: Datos básicos, FastF1 legacy API
- **2023**: Datos mejorados, nuevas métricas
- **2024**: Datos completos, API moderna
- **2025**: Datos actuales, todas las características

### Adaptaciones Automáticas
- **Drivers**: Mapeo automático de códigos entre años
- **Equipos**: Manejo de cambios de nombre (Alfa Romeo → Sauber, etc.)
- **Métricas**: Estimación de características faltantes
- **Normalización**: Comparabilidad entre años diferentes

## ⚠️ Consideraciones Importantes

### Limitaciones por Año
1. **2022**: 
   - Menos métricas avanzadas disponibles
   - Algunos datos estimados

2. **Años Futuros**:
   - Verificación automática de carreras completadas
   - Manejo inteligente de carreras no ocurridas

### Recomendaciones
1. **Primera Ejecución**: Puede tomar más tiempo descargando datos históricos
2. **Cache**: Datos se guardan automáticamente para evitar redescargas
3. **Calidad**: Más años = mejor entrenamiento, pero verificar calidad de datos históricos

## 🔧 Troubleshooting

### Error: "No se pudieron obtener datos para año X"
- **Solución**: Verificar conectividad y API de FastF1
- **Fallback**: Sistema usa datos estimados automáticamente

### Error: "Driver no encontrado en año X"
- **Solución**: Verificar configuración en `config_historical.py`
- **Agregar**: Nuevos drivers al mapeo histórico

### Rendimiento Lento
- **Cache**: Verificar que el cache funcione correctamente
- **Red**: Descargas iniciales pueden ser lentas
- **Datos**: Considerar reducir años si hay problemas

## 📈 Beneficios del Sistema Histórico

1. **Más Datos**: 4 años de datos vs 2 años anteriormente
2. **Mejor Entrenamiento**: Más muestras para ML
3. **Contexto Histórico**: Tendencias de equipos y drivers
4. **Robustez**: Manejo de diferentes formatos de datos
5. **Flexibilidad**: Fácil agregar/quitar años

## 🎯 Próximos Pasos

Para extender aún más el soporte histórico:

1. **Agregar 2021**: Modificar `config_historical.py`
2. **Más Métricas**: Expandir `historical_feature_extractor.py`
3. **Validación**: Agregar tests para datos históricos
4. **Optimización**: Mejorar velocidad de procesamiento

## 📝 Archivos Modificados

```
app/
├── config_historical.py                    # NUEVO - Configuración histórica
├── core/
│   ├── features/
│   │   └── historical_feature_extractor.py # NUEVO - Extractor adaptativo
│   ├── adapters/
│   │   └── progressive_adapter.py          # MODIFICADO - Soporte histórico
│   ├── predictors/
│   │   └── simple_position_predictor.py    # MODIFICADO - Import histórico
│   ├── utils/
│   │   └── race_range_builder.py           # MODIFICADO - Sin hardcode
│   └── pipeline.py                         # MODIFICADO - Collector inteligente
└── data/
    └── collectors/
        └── historical_fastf1_collector.py  # NUEVO - Collector histórico
```
