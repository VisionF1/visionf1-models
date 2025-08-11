import fastf1
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from fastf1 import plotting

class FastF1Collector:
    def __init__(self, race_range):
        self.race_range = race_range
        self.data = []
        self.cache_dir = "app/models_cache/raw_data"
        
        # Crear directorio de cache si no existe
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def collect_data(self):
        """Recolecta datos usando cache inteligente"""
        print(f"🔍 Verificando cache para {len(self.race_range)} carreras...")
        
        fresh_data_collected = 0
        cached_data_used = 0
        
        for race in self.race_range:
            cache_file = self._get_cache_filename(race)
            
            # Intentar cargar desde cache primero
            cached_data = self._load_from_cache(cache_file)
            
            if cached_data is not None and not cached_data.empty:
                self.data.append(cached_data)
                cached_data_used += 1
                print(f"📦 Cache usado para carrera {race['race_name']} ({race['year']})")
            else:
                # Cache no existe o está expirado, descargar datos frescos
                fresh_data = self._download_fresh_data(race)
                if fresh_data is not None and not fresh_data.empty:
                    self.data.append(fresh_data)
                    self._save_to_cache(fresh_data, cache_file, race)
                    fresh_data_collected += 1
                    print(f"🌐 Datos frescos descargados para carrera {race['race_name']} ({race['year']})")
                else:
                    print(f"⚠️  No se pudieron obtener datos para carrera {race['race_name']} ({race['year']})")
        
        print(f"\n📊 Resumen de recolección:")
        print(f"   📦 Datos desde cache: {cached_data_used}")
        print(f"   🌐 Datos descargados: {fresh_data_collected}")
        print(f"   📁 Total carreras procesadas: {len(self.data)}")

    def _get_cache_filename(self, race):
        """Genera nombre de archivo de cache único por carrera"""
        # Sanitizar nombre de carrera para usar en archivo
        race_name = race.get('race_name', f"race_{race.get('round_number', 'unknown')}")
        safe_name = "".join(c for c in race_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        return os.path.join(self.cache_dir, f"race_{race['year']}_{safe_name}_complete.pkl")

    def _load_from_cache(self, cache_file):
        """Carga datos desde cache si existe y no está expirado"""
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Verificar si el archivo es reciente
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            
            # Cargar datos del cache
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verificar que los datos sean válidos
            if isinstance(cache_data, dict) and 'data' in cache_data:
                return cache_data['data']
            
        except Exception as e:
            print(f"❌ Error leyendo cache {os.path.basename(cache_file)}: {e}")
            # Si hay error, eliminar archivo corrupto
            try:
                os.remove(cache_file)
            except:
                pass
        
        return None

    def _save_to_cache(self, data, cache_file, race_info):
        """Guarda datos en cache con metadata"""
        try:
            cache_data = {
                'data': data,
                'metadata': {
                    'cached_at': datetime.now().isoformat(),
                    'race_info': race_info,
                    'data_shape': data.shape if hasattr(data, 'shape') else 'unknown',
                    'drivers_count': len(data) if not data.empty else 0,
                    'includes_practice_quali': True  # Nueva metadata
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"💾 Datos completos guardados en cache: {os.path.basename(cache_file)}")
            
        except Exception as e:
            print(f"❌ Error guardando cache: {e}")

    def _download_fresh_data(self, race):
        """Descarga datos frescos de FastF1 - TODAS LAS SESIONES"""
        try:
            race_identifier = race.get('round_number', race.get('race_name'))
            race_name = race.get('race_name', f'carrera {race_identifier}')
            year = race['year']
            
            print(f"🌐 Descargando datos completos de {race_name} del año {year}...")
            
            # 🔥 RECOLECTAR DATOS DE TODAS LAS SESIONES
            weekend_data = self._extract_complete_weekend_data(year, race_identifier, race_name)
            
            if weekend_data is not None and not weekend_data.empty:
                print(f"✅ Datos completos descargados ({len(weekend_data)} pilotos con práctica/quali/carrera)")
                return weekend_data
            else:
                print(f"⚠️  No se encontraron datos válidos del fin de semana")
                return None
                    
        except Exception as e:
            print(f"❌ Error descargando {race_name}: {e}")
            return None

    def _extract_complete_weekend_data(self, year, race_identifier, race_name):
        """Extrae datos completos del fin de semana: FP1, FP2, FP3, Q, R"""
        try:
            # Definir sesiones a recolectar
            sessions_config = {
                'FP1': 'FP1',
                'FP2': 'FP2', 
                'FP3': 'FP3',
                'Q': 'Q',    # Clasificación completa
                'R': 'R'     # Carrera
            }
            
            weekend_data = {}
            
            for session_name, session_code in sessions_config.items():
                try:
                    print(f"   📊 Extrayendo datos de {session_name}...")
                    session = fastf1.get_session(year, race_identifier, session_code)
                    
                    # Verificar si la sesión ya ocurrió
                    if hasattr(session, 'date') and session.date > pd.Timestamp.now():
                        print(f"   ⏸️  {session_name} aún no ha ocurrido")
                        continue
                    
                    session.load()
                    
                    if session_name == 'Q':
                        # Datos especiales de clasificación
                        session_data = self._extract_qualifying_data(session)
                    elif session_name == 'R':
                        # Datos especiales de carrera
                        session_data = self._extract_race_data(session)
                    else:
                        # Datos de práctica libre
                        session_data = self._extract_practice_data(session, session_name)
                    
                    if session_data:
                        weekend_data[session_name] = session_data
                        print(f"   ✅ {session_name}: {len(session_data)} pilotos")
                    else:
                        print(f"   ⚠️  {session_name}: No hay datos válidos")
                        
                except Exception as e:
                    print(f"   ❌ Error en {session_name}: {e}")
                    continue
            
            # Combinar datos de todas las sesiones por piloto
            return self._combine_weekend_data(weekend_data, race_name, year)
            
        except Exception as e:
            print(f"❌ Error extrayendo datos del fin de semana: {e}")
            return None

    def _extract_qualifying_data(self, session):
        """Extrae datos específicos de clasificación incluyendo condiciones meteorológicas"""
        try:
            qualifying_data = {}
            
            # 🔥 ENSURE SESSION IS LOADED BEFORE ACCESSING DATA
            if not hasattr(session, 'results') or session.results is None:
                print(f"   🔄 Loading qualifying session data...")
                session.load(weather=True)  # Cargar datos meteorológicos
            
            # Obtener condiciones meteorológicas promedio para la sesión
            weather_conditions = self._extract_session_weather_data(session)
            
            # Obtener resultados de clasificación
            if hasattr(session, 'results') and not session.results.empty:
                for _, driver_result in session.results.iterrows():
                    driver = driver_result['Abbreviation']
                    
                    qualifying_data[driver] = {
                        'quali_position': int(driver_result['Position']) if pd.notna(driver_result['Position']) else 20,
                        'q1_time': self._time_to_seconds(driver_result.get('Q1', None)),
                        'q2_time': self._time_to_seconds(driver_result.get('Q2', None)),
                        'q3_time': self._time_to_seconds(driver_result.get('Q3', None)),
                        'quali_best_time': self._time_to_seconds(driver_result.get('Q3', driver_result.get('Q2', driver_result.get('Q1', None)))),
                        'grid_position': int(driver_result['GridPosition']) if pd.notna(driver_result['GridPosition']) else 20,
                        **weather_conditions  # Agregar condiciones meteorológicas
                    }
            
            # También obtener datos de las vueltas de clasificación si no hay resultados
            if not qualifying_data and hasattr(session, 'laps') and not session.laps.empty:
                for driver in session.laps['Driver'].unique():
                    driver_laps = session.laps[session.laps['Driver'] == driver]
                    valid_laps = driver_laps.dropna(subset=['LapTime'])
                    
                    if not valid_laps.empty:
                        best_lap = valid_laps.loc[valid_laps['LapTime'].idxmin()]
                        
                        qualifying_data[driver] = {
                            'quali_position': 20,  # Default if no results available
                            'grid_position': 20,
                            'quali_best_lap_from_laps': best_lap['LapTime'].total_seconds(),
                            'quali_sector1': self._time_to_seconds(best_lap.get('Sector1Time', None)),
                            'quali_sector2': self._time_to_seconds(best_lap.get('Sector2Time', None)),
                            'quali_sector3': self._time_to_seconds(best_lap.get('Sector3Time', None)),
                            **weather_conditions  # Agregar condiciones meteorológicas
                        }
            
            return qualifying_data
            
        except Exception as e:
            print(f"Error extrayendo datos de clasificación: {e}")
            return {}

    def _extract_race_data(self, session):
        """Extrae datos específicos de carrera incluyendo condiciones meteorológicas"""
        try:
            race_data = {}
            
            # Cargar datos meteorológicos si no están cargados
            if not hasattr(session, 'weather_data') or session.weather_data is None:
                print(f"   🔄 Loading race weather data...")
                session.load(weather=True)
            
            # Obtener condiciones meteorológicas promedio para la sesión
            weather_conditions = self._extract_session_weather_data(session)
            
            # Obtener resultados de carrera
            if hasattr(session, 'results') and not session.results.empty:
                for _, driver_result in session.results.iterrows():
                    driver = driver_result['Abbreviation']
                    
                    race_data[driver] = {
                        'race_position': int(driver_result['Position']) if pd.notna(driver_result['Position']) else 20,
                        'points': float(driver_result['Points']) if pd.notna(driver_result['Points']) else 0.0,
                        'race_time': self._time_to_seconds(driver_result.get('Time', None)),
                        'status': driver_result.get('Status', 'Unknown'),
                        **weather_conditions  # Agregar condiciones meteorológicas
                    }
            
            # Obtener datos de vueltas de carrera
            if hasattr(session, 'laps') and not session.laps.empty:
                for driver in session.laps['Driver'].unique():
                    if driver not in race_data:
                        race_data[driver] = {}
                    
                    driver_laps = session.laps[session.laps['Driver'] == driver]
                    valid_laps = driver_laps.dropna(subset=['LapTime'])
                    
                    if not valid_laps.empty:
                        best_lap = valid_laps.loc[valid_laps['LapTime'].idxmin()]
                        clean_air_pace = self.calculate_clean_air_pace(valid_laps)
                        
                        # 🔥 NUEVAS FEATURES DE RENDIMIENTO Y NEUMÁTICOS
                        tyre_data = self._extract_tyre_strategy_data(valid_laps)
                        speed_data = self._extract_speed_profile_data(best_lap, valid_laps)
                        performance_data = self._calculate_performance_metrics(valid_laps, driver)
                        
                        race_data[driver].update({
                            'race_best_lap_time': best_lap['LapTime'].total_seconds(),
                            'race_sector1': self._time_to_seconds(best_lap.get('Sector1Time', None)),
                            'race_sector2': self._time_to_seconds(best_lap.get('Sector2Time', None)),
                            'race_sector3': self._time_to_seconds(best_lap.get('Sector3Time', None)),
                            'clean_air_pace': clean_air_pace,
                            'total_laps': len(valid_laps),
                            **weather_conditions,  # Agregar condiciones meteorológicas si no estaban
                            **tyre_data,  # Estrategia de neumáticos
                            **speed_data,  # Perfil de velocidades
                            **performance_data  # Métricas de rendimiento
                        })
            
            return race_data
            
        except Exception as e:
            print(f"Error extrayendo datos de carrera: {e}")
            return {}

    def _extract_practice_data(self, session, session_name):
        """Extrae datos de sesiones de práctica libre incluyendo condiciones meteorológicas"""
        try:
            practice_data = {}
            
            if not hasattr(session, 'laps') or session.laps is None:
                print(f"   🔄 Loading {session_name} session data...")
                session.load(weather=True)  # Cargar datos meteorológicos
            
            # Obtener condiciones meteorológicas promedio para la sesión
            weather_conditions = self._extract_session_weather_data(session)
            
            if hasattr(session, 'laps') and not session.laps.empty:
                for driver in session.laps['Driver'].unique():
                    driver_laps = session.laps[session.laps['Driver'] == driver]
                    valid_laps = driver_laps.dropna(subset=['LapTime'])
                    
                    if not valid_laps.empty:
                        best_lap = valid_laps.loc[valid_laps['LapTime'].idxmin()]
                        avg_lap = valid_laps['LapTime'].mean()
                        
                        # 🔥 NUEVAS FEATURES PARA PRÁCTICA LIBRE
                        speed_data = self._extract_speed_profile_data(best_lap, valid_laps)
                        consistency_data = self._calculate_consistency_metrics(valid_laps)
                        
                        practice_data[driver] = {
                            f'{session_name.lower()}_best_time': best_lap['LapTime'].total_seconds(),
                            f'{session_name.lower()}_avg_time': avg_lap.total_seconds(),
                            f'{session_name.lower()}_laps_count': len(valid_laps),
                            f'{session_name.lower()}_sector1': self._time_to_seconds(best_lap.get('Sector1Time', None)),
                            f'{session_name.lower()}_sector2': self._time_to_seconds(best_lap.get('Sector2Time', None)),
                            f'{session_name.lower()}_sector3': self._time_to_seconds(best_lap.get('Sector3Time', None)),
                            **weather_conditions,  # Agregar condiciones meteorológicas
                            # Prefijos específicos para práctica libre
                            **{f"{session_name.lower()}_{k}": v for k, v in speed_data.items()},
                            **{f"{session_name.lower()}_{k}": v for k, v in consistency_data.items()}
                        }
            
            return practice_data
            
        except Exception as e:
            print(f"Error extrayendo datos de {session_name}: {e}")
            return {}

    def _combine_weekend_data(self, weekend_data, race_name, year):
        """Combina datos de todas las sesiones del fin de semana"""
        try:
            # Obtener todos los pilotos únicos
            all_drivers = set()
            for session_data in weekend_data.values():
                all_drivers.update(session_data.keys())
            
            combined_data = []
            
            for driver in all_drivers:
                # Crear registro completo del piloto
                driver_record = {
                    'driver': driver,
                    'race_name': race_name,
                    'year': year
                }
                
                # Combinar datos de todas las sesiones
                for session_name, session_data in weekend_data.items():
                    if driver in session_data:
                        driver_record.update(session_data[driver])
                
                # Rellenar valores faltantes con valores por defecto
                driver_record = self._fill_missing_weekend_data(driver_record)
                
                combined_data.append(driver_record)
            
            return pd.DataFrame(combined_data)
            
        except Exception as e:
            print(f"Error combinando datos del fin de semana: {e}")
            return pd.DataFrame()

    def _fill_missing_weekend_data(self, driver_record):
        """Rellena datos faltantes con valores por defecto"""
        # Campos críticos con valores por defecto
        defaults = {
            # Clasificación
            'quali_position': 20,
            'grid_position': 20,
            'quali_best_time': None,
            'q1_time': None,
            'q2_time': None,
            'q3_time': None,
            
            # Carrera
            'race_position': 20,
            'race_best_lap_time': None,
            'clean_air_pace': None,
            'points': 0.0,
            'total_laps': 0,
            
            # Práctica libre
            'fp1_best_time': None,
            'fp2_best_time': None,
            'fp3_best_time': None,
            'fp1_laps_count': 0,
            'fp2_laps_count': 0,
            'fp3_laps_count': 0,
            
            # Datos meteorológicos
            'session_air_temp': None,
            'session_track_temp': None,
            'session_humidity': None,
            'session_pressure': None,
            'session_wind_speed': None,
            'session_wind_direction': None,
            'session_rainfall': False,
            'session_air_temp_min': None,
            'session_air_temp_max': None,
            'session_track_temp_min': None,
            'session_track_temp_max': None,
            
            # Estrategia de neumáticos
            'primary_compound': None,
            'avg_tyre_life': None,
            'fresh_tyre_percentage': None,
            'estimated_pit_stops': 0,
            
            # Perfil de velocidades
            'best_lap_speed_i1': None,
            'best_lap_speed_i2': None,
            'best_lap_speed_fl': None,
            'best_lap_speed_st': None,
            'avg_speed_i1': None,
            'avg_speed_i2': None,
            'avg_speed_fl': None,
            'avg_speed_st': None,
            'max_speed_i1': None,
            'max_speed_i2': None,
            'max_speed_fl': None,
            'max_speed_st': None,
            
            # Métricas de rendimiento
            'lap_time_std': None,
            'lap_time_consistency': None,
            'team': None,
            'avg_position': None,
            'position_changes': None,
            'valid_laps_percentage': 100,
            'sector1_percentage': None,
            'sector2_percentage': None,
            'sector3_percentage': None,
        }
        
        for field, default_value in defaults.items():
            if field not in driver_record or pd.isna(driver_record[field]):
                driver_record[field] = default_value
        
        return driver_record

    def _time_to_seconds(self, time_obj):
        """Convierte tiempo a segundos"""
        if pd.isna(time_obj) or time_obj is None:
            return None
        
        if hasattr(time_obj, 'total_seconds'):
            return time_obj.total_seconds()
        
        return float(time_obj) if isinstance(time_obj, (int, float)) else None

    def _extract_session_weather_data(self, session):
        """Extrae y promedia las condiciones meteorológicas de una sesión"""
        try:
            weather_data = {}
            
            # Verificar si hay datos meteorológicos disponibles
            if hasattr(session, 'weather_data') and not session.weather_data.empty:
                weather_df = session.weather_data
                
                # Calcular promedios y valores representativos de la sesión
                weather_data = {
                    'session_air_temp': float(weather_df['AirTemp'].mean()) if 'AirTemp' in weather_df.columns else None,
                    'session_track_temp': float(weather_df['TrackTemp'].mean()) if 'TrackTemp' in weather_df.columns else None,
                    'session_humidity': float(weather_df['Humidity'].mean()) if 'Humidity' in weather_df.columns else None,
                    'session_pressure': float(weather_df['Pressure'].mean()) if 'Pressure' in weather_df.columns else None,
                    'session_wind_speed': float(weather_df['WindSpeed'].mean()) if 'WindSpeed' in weather_df.columns else None,
                    'session_wind_direction': float(weather_df['WindDirection'].mean()) if 'WindDirection' in weather_df.columns else None,
                    'session_rainfall': bool(weather_df['Rainfall'].any()) if 'Rainfall' in weather_df.columns else False,
                    # Agregar temperaturas mínimas y máximas
                    'session_air_temp_min': float(weather_df['AirTemp'].min()) if 'AirTemp' in weather_df.columns else None,
                    'session_air_temp_max': float(weather_df['AirTemp'].max()) if 'AirTemp' in weather_df.columns else None,
                    'session_track_temp_min': float(weather_df['TrackTemp'].min()) if 'TrackTemp' in weather_df.columns else None,
                    'session_track_temp_max': float(weather_df['TrackTemp'].max()) if 'TrackTemp' in weather_df.columns else None,
                }
                
                print(f"   🌤️  Datos meteorológicos: Aire {weather_data.get('session_air_temp', 'N/A')}°C, Pista {weather_data.get('session_track_temp', 'N/A')}°C, Humedad {weather_data.get('session_humidity', 'N/A')}%, Lluvia {'Sí' if weather_data.get('session_rainfall', False) else 'No'}")
                
            else:
                # Valores por defecto si no hay datos meteorológicos
                weather_data = {
                    'session_air_temp': None,
                    'session_track_temp': None,
                    'session_humidity': None,
                    'session_pressure': None,
                    'session_wind_speed': None,
                    'session_wind_direction': None,
                    'session_rainfall': False,
                    'session_air_temp_min': None,
                    'session_air_temp_max': None,
                    'session_track_temp_min': None,
                    'session_track_temp_max': None,
                }
                print(f"   ⚠️  No hay datos meteorológicos disponibles para esta sesión")
            
            return weather_data
            
        except Exception as e:
            print(f"   ❌ Error extrayendo datos meteorológicos: {e}")
            # Retornar estructura por defecto en caso de error
            return {
                'session_air_temp': None,
                'session_track_temp': None,
                'session_humidity': None,
                'session_pressure': None,
                'session_wind_speed': None,
                'session_wind_direction': None,
                'session_rainfall': False,
                'session_air_temp_min': None,
                'session_air_temp_max': None,
                'session_track_temp_min': None,
                'session_track_temp_max': None,
            }

    def _extract_tyre_strategy_data(self, driver_laps):
        """Extrae datos de estrategia de neumáticos de las vueltas del piloto"""
        try:
            tyre_data = {}
            
            if not driver_laps.empty and 'Compound' in driver_laps.columns:
                # Compuesto más usado
                most_used_compound = driver_laps['Compound'].mode().iloc[0] if not driver_laps['Compound'].mode().empty else None
                
                # Vida del neumático promedio
                avg_tyre_life = driver_laps['TyreLife'].mean() if 'TyreLife' in driver_laps.columns else None
                
                # Porcentaje de vueltas con neumáticos frescos
                fresh_tyre_percentage = (driver_laps['FreshTyre'].sum() / len(driver_laps) * 100) if 'FreshTyre' in driver_laps.columns else None
                
                # Número de paradas (cambios de compuesto)
                pit_stops = len(driver_laps['Compound'].value_counts()) - 1 if 'Compound' in driver_laps.columns else 0
                
                tyre_data = {
                    'primary_compound': most_used_compound,
                    'avg_tyre_life': avg_tyre_life,
                    'fresh_tyre_percentage': fresh_tyre_percentage,
                    'estimated_pit_stops': max(0, pit_stops)
                }
            else:
                tyre_data = {
                    'primary_compound': None,
                    'avg_tyre_life': None,
                    'fresh_tyre_percentage': None,
                    'estimated_pit_stops': 0
                }
            
            return tyre_data
            
        except Exception as e:
            print(f"   ⚠️  Error extrayendo datos de neumáticos: {e}")
            return {
                'primary_compound': None,
                'avg_tyre_life': None,
                'fresh_tyre_percentage': None,
                'estimated_pit_stops': 0
            }

    def _extract_speed_profile_data(self, best_lap, all_laps):
        """Extrae datos de perfil de velocidades"""
        try:
            speed_data = {}
            
            # Velocidades del mejor tiempo
            if hasattr(best_lap, '__getitem__'):
                speed_data['best_lap_speed_i1'] = best_lap.get('SpeedI1', None)
                speed_data['best_lap_speed_i2'] = best_lap.get('SpeedI2', None)
                speed_data['best_lap_speed_fl'] = best_lap.get('SpeedFL', None)
                speed_data['best_lap_speed_st'] = best_lap.get('SpeedST', None)
            
            # Velocidades promedio de todas las vueltas válidas
            if not all_laps.empty:
                speed_data['avg_speed_i1'] = all_laps['SpeedI1'].mean() if 'SpeedI1' in all_laps.columns else None
                speed_data['avg_speed_i2'] = all_laps['SpeedI2'].mean() if 'SpeedI2' in all_laps.columns else None
                speed_data['avg_speed_fl'] = all_laps['SpeedFL'].mean() if 'SpeedFL' in all_laps.columns else None
                speed_data['avg_speed_st'] = all_laps['SpeedST'].mean() if 'SpeedST' in all_laps.columns else None
                
                # Máximas velocidades alcanzadas
                speed_data['max_speed_i1'] = all_laps['SpeedI1'].max() if 'SpeedI1' in all_laps.columns else None
                speed_data['max_speed_i2'] = all_laps['SpeedI2'].max() if 'SpeedI2' in all_laps.columns else None
                speed_data['max_speed_fl'] = all_laps['SpeedFL'].max() if 'SpeedFL' in all_laps.columns else None
                speed_data['max_speed_st'] = all_laps['SpeedST'].max() if 'SpeedST' in all_laps.columns else None
            
            return speed_data
            
        except Exception as e:
            print(f"   ⚠️  Error extrayendo datos de velocidad: {e}")
            return {
                'best_lap_speed_i1': None, 'best_lap_speed_i2': None,
                'best_lap_speed_fl': None, 'best_lap_speed_st': None,
                'avg_speed_i1': None, 'avg_speed_i2': None,
                'avg_speed_fl': None, 'avg_speed_st': None,
                'max_speed_i1': None, 'max_speed_i2': None,
                'max_speed_fl': None, 'max_speed_st': None
            }

    def _calculate_performance_metrics(self, driver_laps, driver_abbrev):
        """Calcula métricas de rendimiento avanzadas"""
        try:
            performance_data = {}
            
            if not driver_laps.empty:
                # Consistencia de tiempos de vuelta
                lap_times_seconds = driver_laps['LapTime'].dt.total_seconds()
                performance_data['lap_time_std'] = lap_times_seconds.std()
                performance_data['lap_time_consistency'] = 1 / (1 + lap_times_seconds.std()) if lap_times_seconds.std() > 0 else 1
                
                # Información del equipo
                if 'Team' in driver_laps.columns:
                    performance_data['team'] = driver_laps['Team'].iloc[0] if not driver_laps['Team'].empty else None
                
                # Posición promedio durante la carrera
                if 'Position' in driver_laps.columns:
                    valid_positions = driver_laps['Position'].dropna()
                    performance_data['avg_position'] = valid_positions.mean() if not valid_positions.empty else None
                    performance_data['position_changes'] = len(valid_positions.unique()) - 1 if len(valid_positions) > 1 else 0
                
                # Porcentaje de vueltas válidas (no eliminadas)
                if 'Deleted' in driver_laps.columns:
                    performance_data['valid_laps_percentage'] = ((~driver_laps['Deleted']).sum() / len(driver_laps) * 100) if 'Deleted' in driver_laps.columns else 100
                else:
                    performance_data['valid_laps_percentage'] = 100
                
                # Tiempos por sector como porcentaje del tiempo total
                if all(col in driver_laps.columns for col in ['Sector1Time', 'Sector2Time', 'Sector3Time']):
                    best_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
                    total_time = best_lap['LapTime'].total_seconds()
                    if total_time > 0:
                        performance_data['sector1_percentage'] = (self._time_to_seconds(best_lap['Sector1Time']) / total_time * 100) if best_lap['Sector1Time'] else None
                        performance_data['sector2_percentage'] = (self._time_to_seconds(best_lap['Sector2Time']) / total_time * 100) if best_lap['Sector2Time'] else None
                        performance_data['sector3_percentage'] = (self._time_to_seconds(best_lap['Sector3Time']) / total_time * 100) if best_lap['Sector3Time'] else None
            
            return performance_data
            
        except Exception as e:
            print(f"   ⚠️  Error calculando métricas de rendimiento: {e}")
            return {
                'lap_time_std': None,
                'lap_time_consistency': None,
                'team': None,
                'avg_position': None,
                'position_changes': None,
                'valid_laps_percentage': 100,
                'sector1_percentage': None,
                'sector2_percentage': None,
                'sector3_percentage': None
            }

    def _calculate_consistency_metrics(self, driver_laps):
        """Calcula métricas de consistencia para sesiones de práctica"""
        try:
            consistency_data = {}
            
            if not driver_laps.empty:
                lap_times_seconds = driver_laps['LapTime'].dt.total_seconds()
                
                # Desviación estándar de tiempos
                consistency_data['time_std'] = lap_times_seconds.std()
                
                # Diferencia entre mejor y peor vuelta
                consistency_data['time_range'] = lap_times_seconds.max() - lap_times_seconds.min()
                
                # Coeficiente de variación (medida de consistencia relativa)
                if lap_times_seconds.mean() > 0:
                    consistency_data['consistency_cv'] = lap_times_seconds.std() / lap_times_seconds.mean()
                else:
                    consistency_data['consistency_cv'] = None
            
            return consistency_data
            
        except Exception as e:
            print(f"   ⚠️  Error calculando consistencia: {e}")
            return {
                'time_std': None,
                'time_range': None,
                'consistency_cv': None
            }

    def calculate_clean_air_pace(self, driver_laps):
        """Calcula el ritmo en aire limpio"""
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
        """Retorna todos los datos combinados"""
        return pd.concat(self.data, ignore_index=True) if self.data else pd.DataFrame()
    
    def clear_cache(self, older_than_days=None):
        """Limpia el cache opcionalmente"""
        if not os.path.exists(self.cache_dir):
            return
        
        files_removed = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    if older_than_days:
                        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_age.days <= older_than_days:
                            continue
                    
                    os.remove(file_path)
                    files_removed += 1
                    print(f"🗑️  Cache eliminado: {filename}")
                    
                except Exception as e:
                    print(f"❌ Error eliminando {filename}: {e}")
        
        print(f"🧹 Cache limpiado: {files_removed} archivos eliminados")
    
    def cache_info(self):
        """Muestra información sobre el cache"""
        if not os.path.exists(self.cache_dir):
            print("📁 No existe directorio de cache")
            return
        
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        if not cache_files:
            print("📁 Cache vacío")
            return
        
        print(f"📁 Cache info: {len(cache_files)} archivos")
        total_size = 0
        
        for filename in cache_files:
            file_path = os.path.join(self.cache_dir, filename)
            try:
                size = os.path.getsize(file_path)
                age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                total_size += size
                
                print(f"   📄 {filename}: {size/1024:.1f}KB, {age.days} días")
                
            except Exception as e:
                print(f"   ❌ Error leyendo {filename}: {e}")
        
        print(f"💾 Tamaño total del cache: {total_size/1024/1024:.2f}MB")