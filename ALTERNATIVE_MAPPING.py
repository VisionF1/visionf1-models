DRIVER_MAPPINGS_SIMPLE = {
    # Solo casos donde realmente hay inconsistencias
    "VERSTAPPEN": "VER",
    "HAMILTON": "HAM", 
    "RUSSELL": "RUS",
    "LECLERC": "LEC",
    "NORRIS": "NOR",
    "ALONSO": "ALO",
    "SARGEANT": "SAR",
    "SERGEANT": "SAR",  # Posible typo
    
}

def map_driver_simple(driver_code):
    """Mapeo simple - solo casos especiales"""
    if not driver_code:
        return "UNK"
    
    normalized = str(driver_code).upper().strip()
    return DRIVER_MAPPINGS_SIMPLE.get(normalized, normalized)
