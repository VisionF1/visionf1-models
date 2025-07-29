import pandas as pd
import numpy as np

class FeatureExtractor:
    """Extrae todas las características disponibles de los datos"""
    
    def extract_all_practice_features(self, data):
        """Extrae TODAS las características de práctica libre disponibles"""
        try:
            practice_features = {}
            sessions = ['fp1', 'fp2', 'fp3']
            
            for session in sessions:
                for metric in ['best_time', 'avg_time', 'laps_count']:
                    col_name = f'{session}_{metric}'
                    if col_name in data.columns:
                        practice_features[col_name] = pd.to_numeric(data[col_name], errors='coerce')
                
                for sector in [1, 2, 3]:
                    sector_col = f'{session}_sector{sector}'
                    if sector_col in data.columns:
                        practice_features[sector_col] = pd.to_numeric(data[sector_col], errors='coerce')
            
            return pd.DataFrame(practice_features) if practice_features else None
            
        except Exception as e:
            print(f"❌ Error extrayendo características de práctica: {e}")
            return None

    def extract_all_qualifying_features(self, data):
        """Extrae TODAS las características de clasificación disponibles"""
        try:
            quali_features = {}
            
            for pos_col in ['quali_position', 'grid_position']:
                if pos_col in data.columns:
                    quali_features[pos_col] = pd.to_numeric(data[pos_col], errors='coerce')
            
            for q_session in ['q1_time', 'q2_time', 'q3_time', 'quali_best_time']:
                if q_session in data.columns:
                    quali_features[q_session] = pd.to_numeric(data[q_session], errors='coerce')
            
            for sector in [1, 2, 3]:
                sector_col = f'quali_sector{sector}'
                if sector_col in data.columns:
                    quali_features[sector_col] = pd.to_numeric(data[sector_col], errors='coerce')
            
            if 'quali_best_lap_from_laps' in data.columns:
                quali_features['quali_best_lap_from_laps'] = pd.to_numeric(data['quali_best_lap_from_laps'], errors='coerce')
            
            return pd.DataFrame(quali_features) if quali_features else None
            
        except Exception as e:
            print(f"❌ Error extrayendo características de clasificación: {e}")
            return None

    def extract_all_race_features(self, data):
        """Extrae TODAS las características de carrera disponibles"""
        try:
            race_features = {}
            
            for col in ['race_position', 'points']:
                if col in data.columns:
                    race_features[col] = pd.to_numeric(data[col], errors='coerce')
            
            for col in ['race_best_lap_time', 'race_time']:
                if col in data.columns:
                    race_features[col] = pd.to_numeric(data[col], errors='coerce')
            
            for sector in [1, 2, 3]:
                sector_col = f'race_sector{sector}'
                if sector_col in data.columns:
                    race_features[sector_col] = pd.to_numeric(data[sector_col], errors='coerce')
            
            for col in ['total_laps', 'status']:
                if col in data.columns:
                    if col == 'status':
                        race_features[f'{col}_numeric'] = data[col].map({
                            'Finished': 1, '+1 Lap': 0.9, '+2 Laps': 0.8, 'DNF': 0, 'DNS': 0
                        }).fillna(0.5)
                    else:
                        race_features[col] = pd.to_numeric(data[col], errors='coerce')
            
            return pd.DataFrame(race_features) if race_features else None
            
        except Exception as e:
            print(f"❌ Error extrayendo características de carrera: {e}")
            return None

    def extract_derived_features(self, data):
        """Extrae características derivadas críticas"""
        try:
            derived_features = {}
            
            # Diferencia quali vs carrera
            if 'quali_position' in data.columns and 'race_position' in data.columns:
                quali_pos = pd.to_numeric(data['quali_position'], errors='coerce')
                race_pos = pd.to_numeric(data['race_position'], errors='coerce')
                derived_features['positions_gained'] = quali_pos - race_pos
            
            # Grid position vs race position
            if 'grid_position' in data.columns and 'race_position' in data.columns:
                grid_pos = pd.to_numeric(data['grid_position'], errors='coerce')
                race_pos = pd.to_numeric(data['race_position'], errors='coerce')
                derived_features['grid_to_race_change'] = grid_pos - race_pos
            
            # Consistencia en práctica
            practice_times = []
            for session in ['fp1_best_time', 'fp2_best_time', 'fp3_best_time']:
                if session in data.columns:
                    practice_times.append(pd.to_numeric(data[session], errors='coerce'))
        
            if len(practice_times) >= 2:
                practice_df = pd.concat(practice_times, axis=1)
                derived_features['practice_consistency'] = practice_df.std(axis=1, skipna=True)
                derived_features['practice_progression'] = practice_times[-1] - practice_times[0]
            
            return pd.DataFrame(derived_features) if derived_features else None
            
        except Exception as e:
            print(f"❌ Error extrayendo características derivadas: {e}")
            return None