"""
Recolector de datos histórico mejorado
Maneja las diferencias en APIs y disponibilidad de datos entre 2022-2025
"""

import fastf1
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from app.config_historical import get_drivers_for_year, TEAM_MAPPINGS
from app.core.features.historical_feature_extractor import HistoricalFeatureExtractor

class HistoricalFastF1Collector:
    """Recolector de datos FastF1 con soporte histórico completo"""
    
    def __init__(self, race_range):
        self.race_range = race_range
        self.data = []
        self.cache_dir = "app/models_cache/raw_data"
        self.feature_extractor = HistoricalFeatureExtractor()
        
        # Crear directorio de cache si no existe
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # APIs y características disponibles por año
        self.year_capabilities = {
            2022: {
                'fastf1_version': 'legacy',
                'session_types': ['FP1', 'FP2', 'FP3', 'Q', 'R'],
                'has_sprint': True,
                'telemetry_available': True,
                'weather_data': False
            },
            2023: {
                'fastf1_version': 'modern', 
                'session_types': ['FP1', 'FP2', 'FP3', 'Q', 'R', 'S'],
                'has_sprint': True,
                'telemetry_available': True,
                'weather_data': True
            },
            2024: {
                'fastf1_version': 'current',
                'session_types': ['FP1', 'FP2', 'FP3', 'Q', 'R', 'S'],
                'has_sprint': True,
                'telemetry_available': True,
                'weather_data': True
            },
            2025: {
                'fastf1_version': 'current',
                'session_types': ['FP1', 'FP2', 'FP3', 'Q', 'R', 'S'],
                'has_sprint': True,
                'telemetry_available': True,
                'weather_data': True
            }
        }

    def collect_data(self):
        """Recolecta datos usando cache inteligente y manejo histórico"""
        print(f"🔍 Verificando cache para {len(self.race_range)} carreras históricas...")
        
        # Debug: Mostrar distribución de años
        years_distribution = {}
        for race in self.race_range:
            year = race['year']
            years_distribution[year] = years_distribution.get(year, 0) + 1
        
        print(f"📅 Distribución de carreras por año:")
        for year, count in sorted(years_distribution.items()):
            print(f"   {year}: {count} carreras")
        
        fresh_data_collected = 0
        cached_data_used = 0
        failed_downloads = 0
        
        for race in self.race_range:
            year = race['year']
            cache_file = self._get_cache_filename(race)
            
            print(f"📅 Procesando {race['race_name']} ({year})...")
            
            # Intentar cargar desde cache primero
            cached_data = self._load_from_cache(cache_file)
            
            if cached_data is not None and not cached_data.empty:
                processed_data = self._process_historical_data(cached_data, year)
                self.data.append(processed_data)
                cached_data_used += 1
                print(f"   📦 Cache usado - {len(processed_data)} filas")
            else:
                # Cache no existe, descargar datos frescos con manejo histórico
                fresh_data = self._download_historical_data(race)
                if fresh_data is not None and not fresh_data.empty:
                    processed_data = self._process_historical_data(fresh_data, year)
                    self.data.append(processed_data)
                    self._save_to_cache(fresh_data, cache_file, race)
                    fresh_data_collected += 1
                    print(f"   🌐 Datos frescos descargados - {len(processed_data)} filas")
                else:
                    print(f"   ⚠️  Fallo al obtener datos")
                    failed_downloads += 1
        
        print(f"\n📊 Resumen de recolección histórica:")
        print(f"   📦 Datos desde cache: {cached_data_used}")
        print(f"   🌐 Datos descargados: {fresh_data_collected}")
        print(f"   ❌ Fallos: {failed_downloads}")
        print(f"   📁 Total carreras procesadas: {len(self.data)}")
        
        # Debug: Verificar datos recolectados
        if self.data:
            total_rows = sum(len(df) for df in self.data if not df.empty)
            print(f"   📊 Total filas de datos: {total_rows}")
        else:
            print(f"   ⚠️  No se recolectaron datos")

    def _download_historical_data(self, race):
        """Descarga datos históricos con manejo específico por año"""
        try:
            year = race['year']
            race_identifier = race.get('round_number', race.get('race_name'))
            race_name = race.get('race_name', f'carrera {race_identifier}')
            
            print(f"   🕐 Descargando datos históricos de {race_name} ({year})...")
            
            # Obtener capacidades del año
            capabilities = self.year_capabilities.get(year, self.year_capabilities[2025])
            
            # Intentar obtener el evento
            try:
                event = fastf1.get_event(year, race_identifier)
            except Exception as e:
                print(f"   ❌ Error obteniendo evento: {e}")
                return self._fallback_historical_data(race)
            
            # Recolectar datos de todas las sesiones disponibles
            weekend_data = self._extract_historical_weekend_data(
                event, year, race_identifier, race_name, capabilities
            )
            
            if weekend_data is not None and not weekend_data.empty:
                print(f"   ✅ Datos históricos descargados ({len(weekend_data)} pilotos)")
                return weekend_data
            else:
                print(f"   ⚠️  Sin datos válidos del fin de semana histórico")
                return self._fallback_historical_data(race)
            
        except Exception as e:
            print(f"   ❌ Error descargando datos históricos: {e}")
            return self._fallback_historical_data(race)

    def _extract_historical_weekend_data(self, event, year, race_identifier, race_name, capabilities):
        """Extrae datos del fin de semana con manejo histórico"""
        try:
            weekend_data = {}
            session_types = capabilities['session_types']
            
            for session_name in session_types:
                try:
                    print(f"     🔄 Cargando {session_name}...")
                    
                    # Obtener la sesión
                    session = fastf1.get_session(year, race_identifier, session_name)
                    
                    # Verificar si la sesión ocurrió (especialmente importante para años pasados)
                    if hasattr(session, 'date'):
                        if year >= 2025 and session.date > pd.Timestamp.now():
                            print(f"     ⏸️  {session_name} aún no ha ocurrido")
                            continue
                    
                    session.load()
                    
                    # Extraer datos específicos de la sesión con manejo histórico
                    if session_name == 'Q':
                        session_data = self._extract_historical_qualifying_data(session, year)
                    elif session_name == 'R':
                        session_data = self._extract_historical_race_data(session, year)
                    elif session_name == 'S':  # Sprint
                        session_data = self._extract_historical_sprint_data(session, year)
                    else:  # FP1, FP2, FP3
                        session_data = self._extract_historical_practice_data(session, session_name, year)
                    
                    if session_data:
                        weekend_data[session_name] = session_data
                        print(f"     ✅ {session_name}: {len(session_data)} pilotos")
                    else:
                        print(f"     ⚠️  {session_name}: Sin datos válidos")
                        
                except Exception as e:
                    print(f"     ❌ Error en {session_name}: {e}")
                    continue
            
            # Combinar datos de todas las sesiones con manejo histórico
            return self._combine_historical_weekend_data(weekend_data, race_name, year)
            
        except Exception as e:
            print(f"   ❌ Error extrayendo datos del fin de semana histórico: {e}")
            return None

    def _extract_historical_qualifying_data(self, session, year):
        """Extrae datos de clasificación con manejo histórico"""
        try:
            qualifying_data = {}
            
            if not hasattr(session, 'results') or session.results is None:
                session.load()
            
            if hasattr(session, 'results') and not session.results.empty:
                for _, driver_result in session.results.iterrows():
                    driver = driver_result['Abbreviation']
                    
                    # Manejo específico por año para diferentes formatos de datos
                    if year <= 2022:
                        qualifying_data[driver] = self._extract_2022_qualifying_format(driver_result)
                    else:
                        qualifying_data[driver] = self._extract_modern_qualifying_format(driver_result)
            
            return qualifying_data
            
        except Exception as e:
            print(f"     ❌ Error extrayendo datos de clasificación histórica: {e}")
            return {}

    def _extract_historical_race_data(self, session, year):
        """Extrae datos de carrera con manejo histórico"""
        try:
            race_data = {}
            
            if hasattr(session, 'results') and not session.results.empty:
                for _, driver_result in session.results.iterrows():
                    driver = driver_result['Abbreviation']
                    
                    # Datos básicos disponibles en todos los años
                    base_data = {
                        'final_position': self._safe_get(driver_result, 'Position', 20),
                        'grid_position': self._safe_get(driver_result, 'GridPosition', 20),
                        'points': self._safe_get(driver_result, 'Points', 0),
                        'status': self._safe_get(driver_result, 'Status', 'Unknown')
                    }
                    
                    # Datos específicos por año
                    if year >= 2023:
                        # Datos más detallados disponibles desde 2023
                        enhanced_data = self._extract_enhanced_race_data(session, driver, year)
                        base_data.update(enhanced_data)
                    else:
                        # Datos básicos para 2022
                        basic_data = self._extract_basic_race_data(session, driver, year)
                        base_data.update(basic_data)
                    
                    race_data[driver] = base_data
            
            return race_data
            
        except Exception as e:
            print(f"     ❌ Error extrayendo datos de carrera histórica: {e}")
            return {}

    def _process_historical_data(self, data, year):
        """Procesa datos históricos para normalización"""
        try:
            # Usar el feature extractor histórico
            processed_data = self.feature_extractor.extract_features_for_year(data, year)
            
            # Normalizar para comparabilidad entre años
            normalized_data = self.feature_extractor.normalize_features_across_years(
                processed_data, target_year=2025
            )
            
            return normalized_data
            
        except Exception as e:
            print(f"   ❌ Error procesando datos históricos: {e}")
            return data

    def _fallback_historical_data(self, race):
        """Fallback para datos históricos cuando falla la descarga"""
        year = race['year']
        race_name = race.get('race_name', 'Unknown')
        
        print(f"   🔄 Usando fallback histórico para {race_name} ({year})")
        
        # Crear datos básicos simulados basados en la configuración histórica
        drivers_config = get_drivers_for_year(year)
        fallback_data = []
        
        for i, (driver, config) in enumerate(drivers_config.items(), 1):
            min_pos, max_pos = config.get('expected_range', (i, i+5))
            estimated_position = (min_pos + max_pos) / 2
            
            fallback_data.append({
                'driver': driver,
                'race_name': race_name,
                'year': year,
                'final_position': estimated_position,
                'grid_position': estimated_position + np.random.uniform(-3, 3),
                'team': config.get('team', 'Unknown'),
                'points': max(0, 25 - (estimated_position - 1) * 2),
                'best_lap_time': 90.0 + np.random.uniform(-2, 2),  # Tiempo estimado
                'laps_completed': np.random.randint(50, 65),
                'status': 'Finished' if i <= 18 else 'DNF'
            })
        
        return pd.DataFrame(fallback_data)

    # Métodos auxiliares para manejo específico por año
    def _extract_2022_qualifying_format(self, driver_result):
        """Formato específico de datos de clasificación 2022"""
        return {
            'quali_position': self._safe_get(driver_result, 'Position', 20),
            'q1_time': self._time_to_seconds(driver_result.get('Q1')),
            'q2_time': self._time_to_seconds(driver_result.get('Q2')),
            'q3_time': self._time_to_seconds(driver_result.get('Q3')),
            'quali_best_time': self._time_to_seconds(
                driver_result.get('Q3') or driver_result.get('Q2') or driver_result.get('Q1')
            )
        }

    def _extract_modern_qualifying_format(self, driver_result):
        """Formato moderno de datos de clasificación (2023+)"""
        base_data = self._extract_2022_qualifying_format(driver_result)
        
        # Agregar datos adicionales disponibles desde 2023
        base_data.update({
            'quali_sector1': self._time_to_seconds(driver_result.get('Sector1Time')),
            'quali_sector2': self._time_to_seconds(driver_result.get('Sector2Time')),
            'quali_sector3': self._time_to_seconds(driver_result.get('Sector3Time'))
        })
        
        return base_data

    def _extract_enhanced_race_data(self, session, driver, year):
        """Datos de carrera mejorados (2023+)"""
        enhanced_data = {}
        
        try:
            driver_laps = session.laps[session.laps['Driver'] == driver]
            valid_laps = driver_laps.dropna(subset=['LapTime'])
            
            if not valid_laps.empty:
                best_lap = valid_laps.loc[valid_laps['LapTime'].idxmin()]
                
                enhanced_data.update({
                    'race_best_lap_time': best_lap['LapTime'].total_seconds(),
                    'race_sector1': self._time_to_seconds(best_lap.get('Sector1Time')),
                    'race_sector2': self._time_to_seconds(best_lap.get('Sector2Time')),
                    'race_sector3': self._time_to_seconds(best_lap.get('Sector3Time')),
                    'clean_air_pace': self._calculate_clean_air_pace(valid_laps),
                    'laps_completed': len(valid_laps)
                })
        
        except Exception as e:
            print(f"       ⚠️  Error extrayendo datos de carrera mejorados: {e}")
        
        return enhanced_data

    def _extract_basic_race_data(self, session, driver, year):
        """Datos básicos de carrera (2022)"""
        basic_data = {}
        
        try:
            driver_laps = session.laps[session.laps['Driver'] == driver]
            valid_laps = driver_laps.dropna(subset=['LapTime'])
            
            if not valid_laps.empty:
                best_lap_time = valid_laps['LapTime'].min()
                basic_data.update({
                    'race_best_lap_time': best_lap_time.total_seconds(),
                    'laps_completed': len(valid_laps),
                    # Estimar clean air pace para 2022
                    'clean_air_pace': best_lap_time.total_seconds() + np.random.uniform(0.5, 1.5)
                })
        
        except Exception as e:
            print(f"       ⚠️  Error extrayendo datos básicos de carrera: {e}")
        
        return basic_data

    def _safe_get(self, data, key, default=None):
        """Obtiene valor de forma segura"""
        try:
            value = data.get(key, default)
            if pd.isna(value):
                return default
            return value
        except:
            return default

    def _time_to_seconds(self, time_obj):
        """Convierte tiempo a segundos con manejo de errores"""
        if time_obj is None or pd.isna(time_obj):
            return None
        try:
            if hasattr(time_obj, 'total_seconds'):
                return time_obj.total_seconds()
            elif isinstance(time_obj, (int, float)):
                return float(time_obj)
            else:
                return None
        except:
            return None

    def _calculate_clean_air_pace(self, driver_laps):
        """Calcula ritmo en aire limpio con manejo histórico"""
        try:
            # Filtrar vueltas en aire limpio (heurística simple)
            clean_laps = driver_laps[driver_laps['LapTime'] < driver_laps['LapTime'].quantile(0.7)]
            if len(clean_laps) >= 3:
                return clean_laps['LapTime'].mean().total_seconds()
            else:
                return driver_laps['LapTime'].min().total_seconds()
        except:
            return None

    # Métodos heredados y adaptados de la clase original
    def _get_cache_filename(self, race):
        """Genera nombre de archivo de cache único por carrera"""
        race_name = race.get('race_name', f"race_{race.get('round_number', 'unknown')}")
        safe_name = "".join(c for c in race_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        return os.path.join(self.cache_dir, f"race_{race['year']}_{safe_name}_historical.pkl")

    def _load_from_cache(self, cache_file):
        """Carga datos desde cache si existe"""
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            if isinstance(cache_data, dict) and 'data' in cache_data:
                return cache_data['data']
            
        except Exception as e:
            print(f"❌ Error leyendo cache {os.path.basename(cache_file)}: {e}")
            try:
                os.remove(cache_file)
            except:
                pass
        
        return None

    def _save_to_cache(self, data, cache_file, race_info):
        """Guarda datos en cache con metadata histórica"""
        try:
            cache_data = {
                'data': data,
                'metadata': {
                    'cached_at': datetime.now().isoformat(),
                    'race_info': race_info,
                    'data_shape': data.shape if hasattr(data, 'shape') else 'unknown',
                    'historical_processing': True,
                    'year': race_info.get('year'),
                    'feature_version': '2.0_historical'
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            print(f"❌ Error guardando cache {os.path.basename(cache_file)}: {e}")

    def get_data(self):
        """Retorna todos los datos combinados"""
        if not self.data:
            print("⚠️  No hay datos para combinar")
            return pd.DataFrame()
        
        try:
            print(f"🔄 Combinando {len(self.data)} DataFrames...")
            
            # Debug: Mostrar info de cada DataFrame antes de combinar
            for i, df in enumerate(self.data):
                if not df.empty:
                    year = df['year'].iloc[0] if 'year' in df.columns else 'Unknown'
                    print(f"   DataFrame {i}: {len(df)} filas, año {year}")
                else:
                    print(f"   DataFrame {i}: VACÍO")
            
            # Filtrar DataFrames vacíos
            non_empty_data = [df for df in self.data if not df.empty]
            
            if not non_empty_data:
                print("❌ Todos los DataFrames están vacíos")
                return pd.DataFrame()
            
            combined_data = pd.concat(non_empty_data, ignore_index=True)
            
            print(f"✅ Datos combinados exitosamente:")
            print(f"   Total filas: {len(combined_data)}")
            
            if 'year' in combined_data.columns:
                year_counts = combined_data['year'].value_counts().sort_index()
                print(f"   Distribución por año:")
                for year, count in year_counts.items():
                    print(f"     {year}: {count} filas")
            
            return combined_data
            
        except Exception as e:
            print(f"❌ Error combinando datos: {e}")
            return pd.DataFrame()

    def _extract_historical_practice_data(self, session, session_name, year):
        """Extrae datos de práctica libre con manejo histórico"""
        try:
            practice_data = {}
            
            if not hasattr(session, 'results') or session.results is None:
                session.load()
            
            if hasattr(session, 'results') and not session.results.empty:
                for _, driver_result in session.results.iterrows():
                    driver = driver_result['Abbreviation']
                    
                    practice_data[driver] = {
                        f'{session_name.lower()}_position': self._safe_get(driver_result, 'Position', 20),
                        f'{session_name.lower()}_best_time': self._time_to_seconds(driver_result.get('Time')),
                        f'{session_name.lower()}_laps': self._safe_get(driver_result, 'LapsCompleted', 0)
                    }
            
            return practice_data
            
        except Exception as e:
            print(f"     ❌ Error extrayendo datos de práctica histórica: {e}")
            return {}

    def _extract_historical_sprint_data(self, session, year):
        """Extrae datos de sprint con manejo histórico"""
        try:
            sprint_data = {}
            
            if hasattr(session, 'results') and not session.results.empty:
                for _, driver_result in session.results.iterrows():
                    driver = driver_result['Abbreviation']
                    
                    sprint_data[driver] = {
                        'sprint_position': self._safe_get(driver_result, 'Position', 20),
                        'sprint_grid': self._safe_get(driver_result, 'GridPosition', 20),
                        'sprint_points': self._safe_get(driver_result, 'Points', 0),
                        'sprint_status': self._safe_get(driver_result, 'Status', 'Unknown')
                    }
            
            return sprint_data
            
        except Exception as e:
            print(f"     ❌ Error extrayendo datos de sprint histórica: {e}")
            return {}

    def _combine_historical_weekend_data(self, weekend_data, race_name, year):
        """Combina datos de todas las sesiones del fin de semana histórico"""
        try:
            if not weekend_data:
                return pd.DataFrame()
            
            # Obtener todos los drivers únicos del fin de semana
            all_drivers = set()
            for session_data in weekend_data.values():
                all_drivers.update(session_data.keys())
            
            combined_records = []
            
            for driver in all_drivers:
                driver_record = {
                    'driver': driver,
                    'race_name': race_name,
                    'year': year
                }
                
                # Combinar datos de todas las sesiones
                for session_name, session_data in weekend_data.items():
                    if driver in session_data:
                        driver_record.update(session_data[driver])
                
                # Rellenar datos faltantes
                driver_record = self._fill_missing_historical_data(driver_record, year)
                combined_records.append(driver_record)
            
            return pd.DataFrame(combined_records)
            
        except Exception as e:
            print(f"   ❌ Error combinando datos del fin de semana histórico: {e}")
            return pd.DataFrame()

    def _fill_missing_historical_data(self, driver_record, year):
        """Rellena datos faltantes con valores por defecto históricos"""
        # Datos básicos por defecto
        defaults = {
            'final_position': 20,
            'grid_position': 20,
            'points': 0,
            'best_lap_time': None,
            'laps_completed': 0,
            'status': 'Unknown'
        }
        
        # Datos de práctica libre
        for fp in ['fp1', 'fp2', 'fp3']:
            defaults.update({
                f'{fp}_position': 20,
                f'{fp}_best_time': None,
                f'{fp}_laps': 0
            })
        
        # Datos de clasificación
        defaults.update({
            'quali_position': 20,
            'q1_time': None,
            'q2_time': None,
            'q3_time': None
        })
        
        # Rellenar valores faltantes
        for key, default_value in defaults.items():
            if key not in driver_record or driver_record[key] is None:
                driver_record[key] = default_value
        
        return driver_record
