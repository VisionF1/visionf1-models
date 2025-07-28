def extract_best_lap_time(data):
    """
    Extract the best lap time from the race data.
    
    Parameters:
    data (DataFrame): The race data containing lap times.

    Returns:
    DataFrame: DataFrame with best lap times by driver.
    """
    if 'best_lap_time' not in data.columns:
        print("Columna 'best_lap_time' no encontrada en los datos")
        return data
    
    # Agrupar por driver y obtener el mejor tiempo
    if 'driver' in data.columns:
        best_times = data.groupby('driver')['best_lap_time'].min().reset_index()
        return best_times
    else:
        return data[['best_lap_time']]

def extract_sector_times(data):
    """
    Extract sector times for each lap from the race data.
    
    Parameters:
    data (DataFrame): The race data containing sector times.

    Returns:
    DataFrame: A DataFrame containing sector times for each driver.
    """
    if 'sector_times' not in data.columns:
        print("Columna 'sector_times' no encontrada en los datos")
        return data
    
    # Los sector_times ya están extraídos en el formato correcto
    if 'driver' in data.columns:
        return data[['driver', 'sector_times']]
    else:
        return data[['sector_times']]

def extract_clean_air_pace(data):
    """
    Calculate the clean air race pace, ignoring laps with traffic.
    
    Parameters:
    data (DataFrame): The race data containing lap times and traffic information.

    Returns:
    DataFrame: DataFrame with clean air pace by driver.
    """
    if 'clean_air_pace' not in data.columns:
        print("Columna 'clean_air_pace' no encontrada en los datos")
        return data
    
    if 'driver' in data.columns:
        return data[['driver', 'clean_air_pace']]
    else:
        return data[['clean_air_pace']]