import fastf1
import pandas as pd
from fastf1 import plotting

class FastF1Collector:
    def __init__(self, race_range):
        self.race_range = race_range
        self.data = []

    def collect_data(self):
        for race in self.race_range:
            try:
                print(f"Recolectando datos de carrera {race['race_name']} del año {race['year']}...")
                session = fastf1.get_session(race['year'], race['race_name'], 'R')
                session.load()
                race_data = self.extract_data(session)
                if not race_data.empty:
                    self.data.append(race_data)
                    print(f"✓ Datos recolectados exitosamente")
                else:
                    print(f"⚠ No se encontraron datos válidos")
            except Exception as e:
                print(f"✗ Error recolectando carrera {race['race_name']}: {e}")

    def extract_data(self, session):
        laps = session.laps
        drivers_data = []

        for driver in laps['Driver'].unique():
            try:
                driver_laps = laps[laps['Driver'] == driver]
                
                # Filtrar laps válidos (sin NaN en LapTime)
                valid_laps = driver_laps.dropna(subset=['LapTime'])
                if valid_laps.empty:
                    continue
                    
                best_lap = valid_laps.loc[valid_laps['LapTime'].idxmin()]
                sector_times = valid_laps[['Sector1Time', 'Sector2Time', 'Sector3Time']].min().to_dict()
                clean_air_pace = self.calculate_clean_air_pace(valid_laps)

                drivers_data.append({
                    'driver': driver,
                    'best_lap_time': best_lap['LapTime'].total_seconds() if pd.notna(best_lap['LapTime']) else None,
                    'sector_times': sector_times,
                    'clean_air_pace': clean_air_pace
                })
            except Exception as e:
                print(f"Error procesando piloto {driver}: {e}")

        return pd.DataFrame(drivers_data)

    def calculate_clean_air_pace(self, driver_laps):
        # Intentar usar columna Position para identificar tráfico
        try:
            # Asumir que posiciones bajas (1-5) tienen menos tráfico
            clean_laps = driver_laps[driver_laps['Position'] <= 5]
            if clean_laps.empty:
                # Si no hay laps en posiciones limpias, usar todos
                clean_laps = driver_laps
            
            valid_times = clean_laps.dropna(subset=['LapTime'])
            if not valid_times.empty:
                return valid_times['LapTime'].mean().total_seconds()
        except:
            # Fallback: usar todos los laps válidos
            valid_times = driver_laps.dropna(subset=['LapTime'])
            if not valid_times.empty:
                return valid_times['LapTime'].mean().total_seconds()
        
        return None

    def get_data(self):
        return pd.concat(self.data, ignore_index=True) if self.data else pd.DataFrame()