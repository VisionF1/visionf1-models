def calculate_clean_air_pace(lap_times, traffic_laps):
    """
    Calculate the clean air race pace by ignoring laps with traffic.
    
    Parameters:
    lap_times (list): A list of lap times for the driver.
    traffic_laps (list): A list of lap indices that were affected by traffic.
    
    Returns:
    float: The average lap time in clean air.
    """
    clean_lap_times = [lap for i, lap in enumerate(lap_times) if i not in traffic_laps]
    
    if not clean_lap_times:
        return None  # No clean laps available
    
    return sum(clean_lap_times) / len(clean_lap_times)

def extract_pace_features(data):
    """
    Extract pace features from the provided data.
    
    Parameters:
    data (DataFrame): A DataFrame containing driver and clean_air_pace columns.
    
    Returns:
    DataFrame: DataFrame with pace features by driver.
    """
    if 'clean_air_pace' not in data.columns:
        print("Columna 'clean_air_pace' no encontrada en los datos")
        return data
    
    if 'driver' in data.columns:
        return data[['driver', 'clean_air_pace']]
    else:
        return data[['clean_air_pace']]