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
        """Extrae datos específicos de clasificación"""
        try:
            qualifying_data = {}
            
            # 🔥 ENSURE SESSION IS LOADED BEFORE ACCESSING DATA
            if not hasattr(session, 'results') or session.results is None:
                print(f"   🔄 Loading qualifying session data...")
                session.load()
            
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
                        'grid_position': int(driver_result['GridPosition']) if pd.notna(driver_result['GridPosition']) else 20
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
                        }
            
            return qualifying_data
            
        except Exception as e:
            print(f"Error extrayendo datos de clasificación: {e}")
            return {}

    def _extract_race_data(self, session):
        """Extrae datos específicos de carrera"""
        try:
            race_data = {}
            
            # Obtener resultados de carrera
            if hasattr(session, 'results') and not session.results.empty:
                for _, driver_result in session.results.iterrows():
                    driver = driver_result['Abbreviation']
                    
                    race_data[driver] = {
                        'race_position': int(driver_result['Position']) if pd.notna(driver_result['Position']) else 20,
                        'points': float(driver_result['Points']) if pd.notna(driver_result['Points']) else 0.0,
                        'race_time': self._time_to_seconds(driver_result.get('Time', None)),
                        'status': driver_result.get('Status', 'Unknown')
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
                        
                        race_data[driver].update({
                            'race_best_lap_time': best_lap['LapTime'].total_seconds(),
                            'race_sector1': self._time_to_seconds(best_lap.get('Sector1Time', None)),
                            'race_sector2': self._time_to_seconds(best_lap.get('Sector2Time', None)),
                            'race_sector3': self._time_to_seconds(best_lap.get('Sector3Time', None)),
                            'clean_air_pace': clean_air_pace,
                            'total_laps': len(valid_laps)
                        })
            
            return race_data
            
        except Exception as e:
            print(f"Error extrayendo datos de carrera: {e}")
            return {}

    def _extract_practice_data(self, session, session_name):
        """Extrae datos de sesiones de práctica libre"""
        try:
            practice_data = {}
            
            if not hasattr(session, 'laps') or session.laps is None:
                print(f"   🔄 Loading {session_name} session data...")
                session.load()
            
            if hasattr(session, 'laps') and not session.laps.empty:
                for driver in session.laps['Driver'].unique():
                    driver_laps = session.laps[session.laps['Driver'] == driver]
                    valid_laps = driver_laps.dropna(subset=['LapTime'])
                    
                    if not valid_laps.empty:
                        best_lap = valid_laps.loc[valid_laps['LapTime'].idxmin()]
                        avg_lap = valid_laps['LapTime'].mean()
                        
                        practice_data[driver] = {
                            f'{session_name.lower()}_best_time': best_lap['LapTime'].total_seconds(),
                            f'{session_name.lower()}_avg_time': avg_lap.total_seconds(),
                            f'{session_name.lower()}_laps_count': len(valid_laps),
                            f'{session_name.lower()}_sector1': self._time_to_seconds(best_lap.get('Sector1Time', None)),
                            f'{session_name.lower()}_sector2': self._time_to_seconds(best_lap.get('Sector2Time', None)),
                            f'{session_name.lower()}_sector3': self._time_to_seconds(best_lap.get('Sector3Time', None))
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
            'fp3_laps_count': 0
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