# Configuración histórica de pilotos por año
# Incluye drivers de 2022, 2023, 2024 y 2025

# Pilotos por año - información histórica completa
DRIVERS_BY_YEAR = {
    2022: {
        # Red Bull Racing - DOMINANTE 🏆
        "VER": {"team": "Red Bull Racing", "tier": 1, "expected_range": (1, 3)},
        "PER": {"team": "Red Bull Racing", "tier": 2, "expected_range": (2, 8)},
        
        # Ferrari - TOP 🥈
        "LEC": {"team": "Ferrari", "tier": 2, "expected_range": (1, 6)},
        "SAI": {"team": "Ferrari", "tier": 2, "expected_range": (3, 10)},
        
        # Mercedes - COMPETITIVO 🥉
        "HAM": {"team": "Mercedes", "tier": 2, "expected_range": (3, 10)},
        "RUS": {"team": "Mercedes", "tier": 2, "expected_range": (4, 12)},
        
        # McLaren - MIDFIELD
        "NOR": {"team": "McLaren", "tier": 3, "expected_range": (6, 15)},
        "RIC": {"team": "McLaren", "tier": 3, "expected_range": (8, 16)},
        
        # Alpine - MIDFIELD
        "OCO": {"team": "Alpine", "tier": 3, "expected_range": (8, 16)},
        "ALO": {"team": "Alpine", "tier": 3, "expected_range": (6, 14)},
        
        # AlphaTauri - MIDFIELD
        "TSU": {"team": "AlphaTauri", "tier": 4, "expected_range": (10, 18)},
        "GAS": {"team": "AlphaTauri", "tier": 4, "expected_range": (8, 16)},
        
        # Aston Martin - MIDFIELD BAJO
        "VET": {"team": "Aston Martin", "tier": 4, "expected_range": (8, 16)},
        "STR": {"team": "Aston Martin", "tier": 4, "expected_range": (12, 18)},
        
        # Williams - BACKMARKERS
        "ALB": {"team": "Williams", "tier": 5, "expected_range": (12, 20)},
        "LAT": {"team": "Williams", "tier": 5, "expected_range": (14, 20)},
        
        # Alfa Romeo - BACKMARKERS
        "BOT": {"team": "Alfa Romeo", "tier": 5, "expected_range": (12, 20)},
        "ZHO": {"team": "Alfa Romeo", "tier": 5, "expected_range": (14, 20)},
        
        # Haas - BACKMARKERS
        "MAG": {"team": "Haas", "tier": 5, "expected_range": (12, 20)},
        "MSC": {"team": "Haas", "tier": 5, "expected_range": (14, 20)}
    },
    
    2023: {
        # Red Bull Racing - DOMINANTE 🏆
        "VER": {"team": "Red Bull Racing", "tier": 1, "expected_range": (1, 2)},
        "PER": {"team": "Red Bull Racing", "tier": 2, "expected_range": (2, 6)},
        
        # Mercedes - TOP 🥈
        "HAM": {"team": "Mercedes", "tier": 2, "expected_range": (2, 8)},
        "RUS": {"team": "Mercedes", "tier": 2, "expected_range": (3, 10)},
        
        # Ferrari - INCONSISTENTE
        "LEC": {"team": "Ferrari", "tier": 2, "expected_range": (3, 10)},
        "SAI": {"team": "Ferrari", "tier": 3, "expected_range": (4, 12)},
        
        # McLaren - MEJORÓ 📈
        "NOR": {"team": "McLaren", "tier": 2, "expected_range": (3, 8)},
        "PIA": {"team": "McLaren", "tier": 3, "expected_range": (6, 12), "rookie": True},
        
        # Aston Martin - TOP 🥈
        "ALO": {"team": "Aston Martin", "tier": 2, "expected_range": (2, 8)},
        "STR": {"team": "Aston Martin", "tier": 3, "expected_range": (6, 14)},
        
        # Alpine - MIDFIELD
        "OCO": {"team": "Alpine", "tier": 3, "expected_range": (8, 16)},
        "GAS": {"team": "Alpine", "tier": 3, "expected_range": (6, 14)},
        
        # Williams - MIDFIELD BAJO
        "ALB": {"team": "Williams", "tier": 4, "expected_range": (10, 18)},
        "SAR": {"team": "Williams", "tier": 4, "expected_range": (12, 20), "rookie": True},
        
        # AlphaTauri - BACKMARKERS 🔻
        "TSU": {"team": "AlphaTauri", "tier": 4, "expected_range": (12, 20)},
        "DEV": {"team": "AlphaTauri", "tier": 5, "expected_range": (15, 20), "rookie": True},
        "RIC": {"team": "AlphaTauri", "tier": 4, "expected_range": (10, 18)},
        
        # Alfa Romeo - BACKMARKERS
        "BOT": {"team": "Alfa Romeo", "tier": 4, "expected_range": (12, 20)},
        "ZHO": {"team": "Alfa Romeo", "tier": 5, "expected_range": (14, 20)},
        
        # Haas - BACKMARKERS
        "MAG": {"team": "Haas", "tier": 4, "expected_range": (12, 20)},
        "HUL": {"team": "Haas", "tier": 4, "expected_range": (10, 18)}
    },
    
    2024: {
        # McLaren - DOMINANTE 🏆
        "NOR": {"team": "McLaren", "tier": 1, "expected_range": (1, 4)},
        "PIA": {"team": "McLaren", "tier": 1, "expected_range": (1, 6)},
        
        # Red Bull Racing - TOP 🥈
        "VER": {"team": "Red Bull Racing", "tier": 1, "expected_range": (1, 4)},
        "PER": {"team": "Red Bull Racing", "tier": 2, "expected_range": (4, 10)},
        
        # Ferrari - TOP 🥈
        "LEC": {"team": "Ferrari", "tier": 2, "expected_range": (2, 8)},
        "SAI": {"team": "Ferrari", "tier": 2, "expected_range": (3, 10)},
        
        # Mercedes - COMPETITIVO 🥉
        "HAM": {"team": "Mercedes", "tier": 2, "expected_range": (3, 10)},
        "RUS": {"team": "Mercedes", "tier": 2, "expected_range": (4, 10)},
        
        # Aston Martin - BAJÓ 📉
        "ALO": {"team": "Aston Martin", "tier": 3, "expected_range": (8, 16)},
        "STR": {"team": "Aston Martin", "tier": 4, "expected_range": (10, 18)},
        
        # RB (ex AlphaTauri) - MIDFIELD
        "TSU": {"team": "RB", "tier": 3, "expected_range": (8, 16)},
        "RIC": {"team": "RB", "tier": 3, "expected_range": (10, 18)},
        
        # Williams  MIDFIELD BAJO
        "ALB": {"team": "Williams", "tier": 4, "expected_range": (12, 20)},
        "SAR": {"team": "Williams", "tier": 4, "expected_range": (12, 20)},
        
        # Alpine - MIDFIELD BAJO
        "OCO": {"team": "Alpine", "tier": 4, "expected_range": (10, 18)},
        "GAS": {"team": "Alpine", "tier": 4, "expected_range": (8, 16)},
        
        # Haas - MIDFIELD BAJO
        "MAG": {"team": "Haas", "tier": 4, "expected_range": (10, 18)},
        "HUL": {"team": "Haas", "tier": 4, "expected_range": (8, 16)},
        
        # Sauber - BACKMARKERS 🔻
        "BOT": {"team": "Sauber", "tier": 5, "expected_range": (14, 20)},
        "ZHO": {"team": "Sauber", "tier": 5, "expected_range": (16, 20)}
    },
    
    2025: {
        # McLaren - DOMINANTE 🏆
        "NOR": {"team": "McLaren", "tier": 1, "expected_range": (1, 4)},
        "PIA": {"team": "McLaren", "tier": 1, "expected_range": (1, 6)},
        
        # Ferrari - TOP 🥈  
        "LEC": {"team": "Ferrari", "tier": 2, "expected_range": (2, 8)},
        "HAM": {"team": "Ferrari", "tier": 2, "expected_range": (3, 10), "team_change": True},
        
        # Red Bull Racing - TOP 🥈
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
}

# Mapeo de drivers solo para casos especiales (cuando hay inconsistencias reales)
# NOTA: Mantener vacío hasta encontrar problemas específicos en los datos
DRIVER_MAPPINGS = {
    # Agregar aquí solo cuando encuentres inconsistencias reales en FastF1:
    # Ejemplo: "VERSTAPPEN": "VER"  # Si FastF1 devuelve nombre completo
    # Ejemplo: "SERGEANT": "SAR"    # Si hay typos en los datos
    
    # Por ahora vacío - FastF1 es bastante consistente con los códigos
}

# Teams que cambiaron de nombre entre años (estos SÍ son necesarios)
TEAM_MAPPINGS = {
    2022: {
        "Alfa Romeo": "Alfa Romeo",
        "AlphaTauri": "AlphaTauri"
    },
    2023: {
        "Alfa Romeo": "Alfa Romeo", 
        "AlphaTauri": "AlphaTauri"
    },
    2024: {
        "Alfa Romeo": "Sauber",
        "AlphaTauri": "RB"
    },
    2025: {
        "Sauber": "Sauber",
        "RB": "Racing Bulls"
    }
}

def get_drivers_for_year(year):
    """Obtiene la configuración de drivers para un año específico"""
    return DRIVERS_BY_YEAR.get(year, DRIVERS_BY_YEAR[2025])

def get_active_drivers_in_year_range(years):
    """Obtiene todos los drivers activos en un rango de años"""
    all_drivers = set()
    for year in years:
        year_drivers = get_drivers_for_year(year)
        all_drivers.update(year_drivers.keys())
    return list(all_drivers)

def map_driver_to_current_format(driver_code, from_year=None):
    """
    Mapea códigos de drivers solo cuando hay inconsistencias conocidas
    
    Args:
        driver_code (str): Código del driver a mapear
        from_year (int, optional): Año de origen (no usado por ahora)
    
    Returns:
        str: Código normalizado del driver
    
    Note:
        Actualmente devuelve el código tal como viene de FastF1.
        Solo mapea si hay casos especiales definidos en DRIVER_MAPPINGS.
    """
    if not driver_code:
        return "UNK"  # Driver desconocido
    
    # Normalizar entrada (mayúsculas, sin espacios)
    normalized_input = str(driver_code).upper().strip()
    
    # Mapear solo si hay casos especiales definidos
    return DRIVER_MAPPINGS.get(normalized_input, normalized_input)

def get_team_name_for_year(team, year):
    """Obtiene el nombre correcto del equipo para un año específico"""
    year_mappings = TEAM_MAPPINGS.get(year, {})
    return year_mappings.get(team, team)
