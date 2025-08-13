# Configuración del rango de carreras para el entrenamiento - MÚLTIPLES AÑOS
RACE_RANGE = {
    "years": [2022, 2023, 2024, 2025],  # Años a descargar
    "max_races_per_year": 24,  # Máximo de carreras por año (F1 tiene ~24 carreras)
    "include_current_year": True,  # Incluir año actual aunque esté incompleto
    "auto_detect_available": True,  # Detectar automáticamente carreras disponibles
    "stop_on_future_races": True   # Parar cuando encuentre carreras futuras
}

# Configuración para predicción de próxima carrera
PREDICTION_CONFIG = {
    "next_race": {
        "year": 2025,
        "race_name": "Hungarian Grand Prix", 
        "circuit_name": "Hungaroring",
        "race_number": 13  # Número de carrera en la temporada 2025
    },
    "use_historical_data": True,
    
    # 🌤️ CONFIGURACIÓN METEOROLÓGICA PARA PREDICCIONES
    "weather_scenarios": {
        
        # Escenario seco - condiciones ideales
        "dry": {
            "session_air_temp": 26.0,      # Temperatura ideal
            "session_track_temp": 35.0,    # Temperatura de pista normal
            "session_humidity": 45.0,      # Humedad baja
            "session_rainfall": False,     # Sin lluvia
            "description": "Condiciones secas e ideales"
        },
        
        # Escenario caluroso - estrés térmico
        "hot": {
            "session_air_temp": 35.0,      # Muy caluroso
            "session_track_temp": 50.0,    # Pista muy caliente
            "session_humidity": 70.0,      # Humedad alta = más estrés
            "session_rainfall": False,     # Sin lluvia
            "description": "Condiciones muy calurosas (estrés térmico)"
        },
        
        # Escenario húmedo - lluvia ligera
        "wet": {
            "session_air_temp": 18.0,      # Más fresco por lluvia
            "session_track_temp": 22.0,    # Pista fría
            "session_humidity": 85.0,      # Muy húmedo
            "session_rainfall": True,      # Lluvia confirmada
            "description": "Condiciones húmedas con lluvia"
        },
        
        # Escenario extremo - tormenta
        "storm": {
            "session_air_temp": 15.0,      # Frío
            "session_track_temp": 18.0,    # Pista muy fría
            "session_humidity": 95.0,      # Humedad extrema
            "session_rainfall": True,      # Lluvia intensa
            "description": "Condiciones extremas - tormenta"
        },
        
        # Escenario frío - condiciones invernales
        "cold": {
            "session_air_temp": 12.0,      # Muy frío
            "session_track_temp": 15.0,    # Pista fría
            "session_humidity": 60.0,      # Humedad media
            "session_rainfall": False,     # Seco pero frío
            "description": "Condiciones muy frías"
        }
    },
    
    # 🎯 CONFIGURACIÓN DE PREDICCIÓN ACTIVA
    "active_scenario": "dry",  # Cambiar por: "dry", "hot", "wet", "storm", "cold"

}

# 🔥 FACTOR DE IMPORTANCIA DE DATOS (SIMPLE)
DATA_IMPORTANCE = {

    "2025_weight": 0.50,  # 50% importancia a datos de 2025
    "2024_weight": 0.25,  # 25% importancia a datos de 2024
    "2023_weight": 0.15,  # 15% importancia a datos de 2023
    "2022_weight": 0.10,  # 10% importancia a datos de 2022

    "ml_vs_config": {
        "ml_weight": 0.80,     # 80% modelo ML (histórico) - MÁS LIBERTAD
        "config_weight": 0.20   # 20% configuración 2025 (actual) - MENOS RESTRICCIÓN
    }
}

# Solo pilotos activos 2025 - SIN TIERS HARDCODEADOS
DRIVERS_2025 = {
    # McLaren 🏆
    "NOR": {"team": "McLaren", "expected_range": (1, 4)},
    "PIA": {"team": "McLaren", "expected_range": (1, 6)},
    
    # Ferrari 🥈  
    "LEC": {"team": "Ferrari", "expected_range": (2, 8)},
    "HAM": {"team": "Ferrari", "expected_range": (3, 10), "team_change": True},
    
    # Red Bull 🥈
    "VER": {"team": "Red Bull Racing", "expected_range": (1, 6)},
    "TSU": {"team": "Red Bull Racing", "expected_range": (8, 15), "team_change": True},
    
    # Mercedes 🥉
    "RUS": {"team": "Mercedes", "expected_range": (4, 10)},
    "ANT": {"team": "Mercedes", "expected_range": (10, 16), "rookie": True},
    
    # Williams 📈
    "ALB": {"team": "Williams", "expected_range": (8, 14)},
    "SAI": {"team": "Williams", "expected_range": (10, 16), "team_change": True},
    
    # Racing Bulls
    "HAD": {"team": "Racing Bulls", "expected_range": (12, 18), "rookie": True},
    "LAW": {"team": "Racing Bulls", "expected_range": (14, 20), "rookie": True},
    
    # Aston Martin 📉
    "ALO": {"team": "Aston Martin", "expected_range": (8, 16)},
    "STR": {"team": "Aston Martin", "expected_range": (12, 18)},
    
    # Haas
    "OCO": {"team": "Haas", "expected_range": (10, 18), "team_change": True},
    "BEA": {"team": "Haas", "expected_range": (15, 20), "rookie": True},
    
    # Alpine 🔻
    "GAS": {"team": "Alpine", "expected_range": (12, 20)},
    "COL": {"team": "Alpine", "expected_range": (14, 20), "team_change": True},
    
    # Sauber 🔻
    "HUL": {"team": "Sauber", "expected_range": (14, 20)},
    "BOR": {"team": "Sauber", "expected_range": (16, 20), "rookie": True}
}

# 🔥 PENALIZACIONES SIMPLES
PENALTIES = {
    "rookie": 2.5,           # Penalización para rookies
    "team_change": 1.5,      # Penalización por cambio de equipo
    "adaptation_races": 8,   # Carreras para adaptarse completamente
    "use_progressive": True  # Usar sistema de adaptación progresiva
}

# Listas simples
ROOKIES_2025 = ["ANT", "BEA", "BOR", "HAD", "LAW"]
RETIRED_DRIVERS = ["PER", "MAG", "DOO", "RIC", "BOT", "ZHO", "SAR"]

# 🔥 CONFIGURACIÓN SIMPLE DE ADAPTACIÓN
ADAPTATION_SYSTEM = {
    "change_types": {
        "rookie": {
            "base_penalty": PENALTIES["rookie"],
            "adaptation_races": PENALTIES["adaptation_races"],
            "description": "Piloto completamente nuevo"
        },
        "team_change": {
            "base_penalty": PENALTIES["team_change"],
            "adaptation_races": 5,
            "description": "Cambio de equipo"
        }
    }
}

# 🔥 FACTORES DE AJUSTE CONSOLIDADOS
ADJUSTMENT_FACTORS = {
    "use_progressive_adaptation": PENALTIES["use_progressive"]
}