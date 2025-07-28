def extract_sector_times(data):
    """
    Extract sector times from the provided race data.

    Parameters:
    data (DataFrame): The race data containing sector times.

    Returns:
    DataFrame: DataFrame with sector times by driver.
    """
    if 'sector_times' not in data.columns:
        print("Columna 'sector_times' no encontrada en los datos")
        return data
    
    if 'driver' in data.columns:
        return data[['driver', 'sector_times']]
    else:
        return data[['sector_times']]

def calculate_average_sector_times(data):
    """
    Calculate average sector times for each driver from the sector_times column.

    Parameters:
    data (DataFrame): DataFrame containing driver and sector_times columns.

    Returns:
    DataFrame: DataFrame with average sector times by driver.
    """
    if 'sector_times' not in data.columns or 'driver' not in data.columns:
        print("Columnas necesarias no encontradas para calcular promedios")
        return data
    
    # Los datos ya contienen los tiempos promedio por sector en el diccionario
    return data[['driver', 'sector_times']]