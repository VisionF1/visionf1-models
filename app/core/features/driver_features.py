def extract_driver_features(data):
    """
    Extrae características relacionadas con los pilotos a partir de los datos de carrera.

    Args:
        data (DataFrame): DataFrame que contiene los datos de carrera.

    Returns:
        DataFrame: DataFrame con las características extraídas de los pilotos.
    """
    if 'driver' not in data.columns:
        print("Columna 'driver' no encontrada en los datos")
        return data
    
    # Usar la columna 'driver' que existe en lugar de 'driverCode' y 'driverName'
    driver_features = data[['driver']].drop_duplicates()
    return driver_features.reset_index(drop=True)

def get_best_lap_time(data):
    """
    Obtiene el mejor tiempo de vuelta por piloto.

    Args:
        data (DataFrame): DataFrame que contiene los datos de carrera.

    Returns:
        DataFrame: DataFrame con el mejor tiempo de vuelta por piloto.
    """
    if 'driver' not in data.columns or 'best_lap_time' not in data.columns:
        print("Columnas necesarias no encontradas en los datos")
        return data
    
    # Usar las columnas que existen
    best_lap_times = data.groupby('driver')['best_lap_time'].min().reset_index()
    return best_lap_times

def get_clean_air_race_pace(data):
    """
    Calcula el ritmo en aire limpio, ignorando las vueltas detrás de otros autos.

    Args:
        data (DataFrame): DataFrame que contiene los datos de carrera.

    Returns:
        DataFrame: DataFrame con el ritmo en aire limpio por piloto.
    """
    if 'driver' not in data.columns or 'clean_air_pace' not in data.columns:
        print("Columnas necesarias no encontradas en los datos")
        return data
    
    # Usar las columnas que existen
    pace = data.groupby('driver')['clean_air_pace'].mean().reset_index()
    pace.columns = ['driver', 'cleanAirPace']
    return pace