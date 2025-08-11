# Configuración del rango de carreras para el entrenamiento - MÚLTIPLES AÑOS
RACE_RANGE = {
    "years": [2022,2023,2024,2025],  # Años a descargar
    "max_races_per_year": 24,  # Máximo de carreras por año (F1 tiene ~24 carreras)
    "include_current_year": True,  # Incluir año actual aunque esté incompleto
    "auto_detect_available": True,  # Detectar automáticamente carreras disponibles
    "stop_on_future_races": True   # Parar cuando encuentre carreras futuras
}

# Configuración para predicción de próxima carrera
PREDICTION_CONFIG = {
    "next_race": {
        "year": 2025,
        "race_name": "Qatar Grand Prix", 
        "circuit_name": "Losail International Circuit",
        "race_number": 23
    },  
    "use_historical_data": False
}

# 🔥 FACTOR DE IMPORTANCIA DE DATOS (SIMPLE)
DATA_IMPORTANCE = {
    "2025_weight": 0.50,  # 50% importancia a datos de 2025
    "2024_weight": 0.25,  # 25% importancia a datos de 2024
    "2023_weight": 0.15,  # 15% importancia a datos de 2023
    "2022_weight": 0.10,   # 10% importancia a datos de 2022
    "ml_vs_config": {
        "ml_weight": 0.50,     # 25% modelo ML (histórico)
        "config_weight": 0.50  # 75% configuración 2025 (actual)
    }
}

# Solo pilotos activos 2025
DRIVERS_2025 = {
    # McLaren - DOMINANTE 🏆
    "NOR": {"team": "McLaren", "tier": 1, "expected_range": (1, 4)},
    "PIA": {"team": "McLaren", "tier": 1, "expected_range": (1, 6)},
    
    # Ferrari - TOP 🥈  
    "LEC": {"team": "Ferrari", "tier": 2, "expected_range": (2, 8)},
    "HAM": {"team": "Ferrari", "tier": 2, "expected_range": (3, 10), "team_change": True},
    
    # Red Bull - TOP 🥈
    "VER": {"team": "Red Bull Racing", "tier": 2, "expected_range": (1, 6)},
    "TSU": {"team": "Red Bull Racing", "tier": 2, "expected_range": (8, 15), "team_change": True},
    
    # Mercedes - COMPETITIVO 🥉
    "RUS": {"team": "Mercedes", "tier": 2, "expected_range": (4, 10)},
    "ANT": {"team": "Mercedes", "tier": 2, "expected_range": (10, 16), "rookie": True},
    
    # Williams - MEJORÓ 📈
    "ALB": {"team": "Williams", "tier": 3, "expected_range": (8, 14)},
    "SAI": {"team": "Williams", "tier": 3, "expected_range": (10, 16), "team_change": True},
    
    # Racing Bulls - MIDFIELD 
    "HAD": {"team": "Racing Bulls", "tier": 3, "expected_range": (12, 18), "rookie": True},
    "LAW": {"team": "Racing Bulls", "tier": 3, "expected_range": (14, 20), "rookie": True},
    
    # Aston Martin - BAJÓ 📉
    "ALO": {"team": "Aston Martin", "tier": 4, "expected_range": (8, 16)},
    "STR": {"team": "Aston Martin", "tier": 4, "expected_range": (12, 18)},
    
    # Haas - MIDFIELD BAJO
    "OCO": {"team": "Haas", "tier": 4, "expected_range": (10, 18), "team_change": True},
    "BEA": {"team": "Haas", "tier": 4, "expected_range": (15, 20), "rookie": True},
    
    # Alpine - BACKMARKERS 🔻
    "GAS": {"team": "Alpine", "tier": 5, "expected_range": (12, 20)},
    "COL": {"team": "Alpine", "tier": 5, "expected_range": (14, 20), "team_change": True},
    
    # Sauber - BACKMARKERS 🔻
    "HUL": {"team": "Sauber", "tier": 5, "expected_range": (14, 20)},
    "BOR": {"team": "Sauber", "tier": 5, "expected_range": (16, 20), "rookie": True}
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
