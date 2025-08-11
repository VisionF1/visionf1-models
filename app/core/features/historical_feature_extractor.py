"""
Sistema de extracción de características histórico-adaptativo
Maneja diferencias en características disponibles entre años 2022-2025
"""

import pandas as pd
import numpy as np
from app.config_historical import get_drivers_for_year, map_driver_to_current_format

class HistoricalFeatureExtractor:
    """Extractor de características que se adapta a diferentes años"""
    
    def __init__(self):
        # Características básicas disponibles en todos los años
        self.core_features = [
            'driver', 'race_name', 'year', 'final_position', 'grid_position',
            'best_lap_time', 'points', 'laps_completed'
        ]
        
        # Características que pueden no estar disponibles en años anteriores
        self.enhanced_features = [
            'fp1_best_time', 'fp2_best_time', 'fp3_best_time',
            'q1_time', 'q2_time', 'q3_time', 'quali_position',
            'race_sector1', 'race_sector2', 'race_sector3',
            'clean_air_pace', 'fp_consistency'
        ]
        
        # Mappings de características por año
        self.feature_mappings = {
            2022: {
                'has_detailed_practice': True,  # FP1, FP2, FP3 disponibles
                'has_sector_data': True,        # Sector times disponibles
                'has_clean_air_pace': False,    # Métrica no calculada en 2022
                'has_sprint_data': True         # Sprint races disponibles
            },
            2023: {
                'has_detailed_practice': True,
                'has_sector_data': True,
                'has_clean_air_pace': True,     # Métrica introducida en 2023
                'has_sprint_data': True
            },
            2024: {
                'has_detailed_practice': True,
                'has_sector_data': True,
                'has_clean_air_pace': True,
                'has_sprint_data': True
            },
            2025: {
                'has_detailed_practice': True,
                'has_sector_data': True,
                'has_clean_air_pace': True,
                'has_sprint_data': True
            }
        }
    
    def extract_features_for_year(self, data, year):
        """Extrae características adaptándose al año específico"""
        if data.empty:
            return self._create_empty_features_dataframe()
        
        year_capabilities = self.feature_mappings.get(year, self.feature_mappings[2025])
        extracted_features = []
        
        for _, row in data.iterrows():
            feature_row = self._extract_core_features(row, year)
            
            # Agregar características avanzadas según disponibilidad del año
            if year_capabilities['has_detailed_practice']:
                feature_row.update(self._extract_practice_features(row))
            
            if year_capabilities['has_sector_data']:
                feature_row.update(self._extract_sector_features(row))
            
            if year_capabilities['has_clean_air_pace']:
                feature_row.update(self._extract_pace_features(row))
            else:
                # Estimar clean air pace para años anteriores
                feature_row.update(self._estimate_clean_air_pace(row))
            
            extracted_features.append(feature_row)
        
        return pd.DataFrame(extracted_features)
    
    def _extract_core_features(self, row, year):
        """Extrae características básicas disponibles en todos los años"""
        # Mapear driver code si es necesario
        driver = map_driver_to_current_format(row.get('driver', 'UNK'), year)
        
        return {
            'driver': driver,
            'race_name': row.get('race_name', 'Unknown'),
            'year': year,
            'final_position': self._safe_numeric(row.get('final_position'), 20),
            'grid_position': self._safe_numeric(row.get('grid_position'), 20),
            'best_lap_time': self._safe_numeric(row.get('best_lap_time')),
            'points': self._safe_numeric(row.get('points'), 0),
            'laps_completed': self._safe_numeric(row.get('laps_completed'), 0),
            'positions_gained': self._calculate_positions_gained(row)
        }
    
    def _extract_practice_features(self, row):
        """Extrae características de práctica libre"""
        return {
            'fp1_best_time': self._safe_numeric(row.get('fp1_best_time')),
            'fp2_best_time': self._safe_numeric(row.get('fp2_best_time')),
            'fp3_best_time': self._safe_numeric(row.get('fp3_best_time')),
            'fp_avg_position': self._calculate_avg_fp_position(row),
            'fp_consistency': self._calculate_fp_consistency(row)
        }
    
    def _extract_sector_features(self, row):
        """Extrae características de sectores"""
        return {
            'quali_sector1': self._safe_numeric(row.get('quali_sector1')),
            'quali_sector2': self._safe_numeric(row.get('quali_sector2')),
            'quali_sector3': self._safe_numeric(row.get('quali_sector3')),
            'race_sector1': self._safe_numeric(row.get('race_sector1')),
            'race_sector2': self._safe_numeric(row.get('race_sector2')),
            'race_sector3': self._safe_numeric(row.get('race_sector3'))
        }
    
    def _extract_pace_features(self, row):
        """Extrae características de ritmo avanzadas"""
        return {
            'clean_air_pace': self._safe_numeric(row.get('clean_air_pace')),
            'race_pace_rank': self._safe_numeric(row.get('race_pace_rank')),
            'quali_performance': self._calculate_quali_performance(row)
        }
    
    def _estimate_clean_air_pace(self, row):
        """Estima clean air pace para años donde no está disponible"""
        best_lap = self._safe_numeric(row.get('best_lap_time'))
        if best_lap is not None:
            # Estimación simple basada en mejor vuelta + factor aleatorio
            estimated_pace = best_lap + np.random.uniform(-1.0, 1.0)
        else:
            estimated_pace = None
        
        return {
            'clean_air_pace': estimated_pace,
            'race_pace_rank': None,  # No estimamos ranking
            'quali_performance': self._calculate_quali_performance(row)
        }
    
    def _calculate_positions_gained(self, row):
        """Calcula posiciones ganadas/perdidas"""
        grid = self._safe_numeric(row.get('grid_position'), 20)
        final = self._safe_numeric(row.get('final_position'), 20)
        if grid is not None and final is not None:
            return grid - final  # Positivo = ganó posiciones
        return 0
    
    def _calculate_avg_fp_position(self, row):
        """Calcula posición promedio en práctica libre"""
        fp_times = [
            row.get('fp1_best_time'),
            row.get('fp2_best_time'),
            row.get('fp3_best_time')
        ]
        valid_times = [t for t in fp_times if t is not None]
        if valid_times:
            return np.mean(valid_times)
        return None
    
    def _calculate_fp_consistency(self, row):
        """Calcula consistencia en práctica libre"""
        fp_times = [
            row.get('fp1_best_time'),
            row.get('fp2_best_time'),
            row.get('fp3_best_time')
        ]
        valid_times = [t for t in fp_times if t is not None and t > 0]
        if len(valid_times) >= 2:
            return np.std(valid_times)
        return None
    
    def _calculate_quali_performance(self, row):
        """Calcula rendimiento en clasificación"""
        q_times = [
            row.get('q1_time'),
            row.get('q2_time'),
            row.get('q3_time')
        ]
        valid_times = [t for t in q_times if t is not None and t > 0]
        if valid_times:
            return min(valid_times)  # Mejor tiempo de clasificación
        return None
    
    def _safe_numeric(self, value, default=None):
        """Convierte valor a numérico de forma segura"""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                if value.lower() in ['', 'nan', 'null', 'none']:
                    return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _create_empty_features_dataframe(self):
        """Crea DataFrame vacío con las columnas esperadas"""
        columns = (self.core_features + 
                  ['positions_gained', 'fp1_best_time', 'fp2_best_time', 'fp3_best_time',
                   'fp_avg_position', 'fp_consistency', 'quali_sector1', 'quali_sector2',
                   'quali_sector3', 'race_sector1', 'race_sector2', 'race_sector3',
                   'clean_air_pace', 'race_pace_rank', 'quali_performance'])
        
        return pd.DataFrame(columns=columns)
    
    def normalize_features_across_years(self, features_df, target_year=2025):
        """Normaliza características para que sean comparables entre años"""
        if features_df.empty:
            return features_df
        
        # Normalizar drivers a formato actual
        features_df['driver'] = features_df.apply(
            lambda row: map_driver_to_current_format(row['driver'], row['year']), 
            axis=1
        )
        
        # Agrupar por año para normalización temporal
        normalized_data = []
        
        for year, year_data in features_df.groupby('year'):
            year_normalized = year_data.copy()
            
            # Normalizar tiempos por circuito/año (para comparabilidad)
            time_columns = ['best_lap_time', 'fp1_best_time', 'fp2_best_time', 'fp3_best_time']
            for col in time_columns:
                if col in year_normalized.columns:
                    year_normalized[f'{col}_normalized'] = self._normalize_times_by_race(
                        year_normalized, col
                    )
            
            normalized_data.append(year_normalized)
        
        return pd.concat(normalized_data, ignore_index=True) if normalized_data else features_df
    
    def _normalize_times_by_race(self, data, time_column):
        """Normaliza tiempos por carrera para comparabilidad"""
        normalized_times = []
        
        for race_name, race_data in data.groupby('race_name'):
            race_times = race_data[time_column].dropna()
            if len(race_times) > 0:
                # Normalizar respecto al tiempo más rápido de la carrera
                fastest_time = race_times.min()
                normalized = race_data[time_column] / fastest_time
            else:
                normalized = race_data[time_column]
            
            normalized_times.extend(normalized.tolist())
        
        return normalized_times
