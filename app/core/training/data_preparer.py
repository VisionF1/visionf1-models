import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from app.core.features.feature_extractor import FeatureExtractor
from app.core.utils.team_mapping_utils import quick_team_mapping

class DataPreparer:
    """Prepara datos para entrenamiento de modelos ML"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor()
        self.feature_names = []
    
    def prepare_training_data(self, data):
        """Prepara datos completos para entrenamiento"""
        print("🔄 Preparando datos para entrenamiento...")
        
        if data is None or data.empty:
            print("❌ No hay datos disponibles para preparar")
            return None, None, None, None, None
        
        training_data = data.copy()
        print(f"📊 Datos originales: {training_data.shape}")
        
        # 🔍 DEBUG: Mostrar columnas disponibles
        print(f"🔍 Columnas disponibles en los datos:")
        columns_list = list(training_data.columns)
        for i, col in enumerate(columns_list):
            print(f"   {i+1:2d}. {col}")
        
        # Buscar columnas relacionadas con equipos
        team_related_columns = [col for col in columns_list if 'team' in col.lower()]
        if team_related_columns:
            print(f"🏎️  Columnas relacionadas con equipos: {team_related_columns}")
        else:
            print(f"⚠️  No se encontraron columnas con 'team' en el nombre")
        
        # 🔧 APLICAR MAPEO DE EQUIPOS ANTES DE ENTRENAR
        print("🔄 Aplicando mapeo histórico de equipos...")
        if 'team' in training_data.columns:
            training_data = quick_team_mapping(training_data)
            unique_teams_after = training_data['team'].dropna().unique()
            print(f"✅ Equipos después del mapeo: {len(unique_teams_after)} equipos únicos")
            print(f"   🏎️ Equipos: {', '.join(sorted(unique_teams_after))}")
            
            # Verificar cuántos valores nulos hay
            null_teams = training_data['team'].isnull().sum()
            if null_teams > 0:
                print(f"⚠️  {null_teams} registros sin información de equipo")
        else:
            print("⚠️ No se encontró columna 'team' en los datos")
            # Buscar alternativas
            possible_team_cols = [col for col in training_data.columns if any(word in col.lower() for word in ['equipo', 'constructor', 'team'])]
            if possible_team_cols:
                print(f"🔍 Posibles columnas de equipo alternativas: {possible_team_cols}")
            else:
                print("❌ No se encontraron columnas de equipo alternativas")
        
        # 1. Preparar características base (8 nuevas características)
        X_base, feature_names = self._prepare_base_features(training_data)
        if X_base is None:
            return None, None, None, None, None
        
        X_df = pd.DataFrame(X_base, columns=feature_names)
        
        # 2. Agregar características adicionales
        X_df = self._add_additional_features(X_df, training_data)
        
        # 3. Preparar variable objetivo ANTES de limpiar
        y = self._prepare_target_variable(training_data)
        if y is None:
            return None, None, None, None, None
        
        # 4. Asegurar misma longitud entre X y y
        min_length = min(len(X_df), len(y))
        X_df = X_df.iloc[:min_length]
        y = y.iloc[:min_length]
        training_data = training_data.iloc[:min_length]
        
        print(f"📏 Sincronizando datos: X{X_df.shape}, y{len(y)}")
        
        # 5. Limpiar datos
        X_clean, training_data_clean = self._clean_data(X_df, training_data, feature_names)
        
        # 6. Filtrar datos válidos
        X_final, y_final = self._filter_valid_data(X_clean, y)
        
        # 6. Dividir en entrenamiento y test
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42
        )
        
        print(f"📊 División de datos completada:")
        print(f"   🎯 Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   🧪 Test: {X_test.shape[0]} muestras")
        print(f"   🔢 Total características: {X_train.shape[1]}")
        
        # Guardar nombres de características
        self.feature_names = list(X_train.columns)
        
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def _prepare_base_features(self, data):
        """Prepara las 8 características base principales"""
        try:
            # Lista de las 8 características principales que necesitamos
            required_features = [
                'team_encoded',
                'team_avg_position_2024',
                'team_avg_position_2023', 
                'team_avg_position_2022',
                'grid_position',
                'quali_best_time',
                'race_best_lap_time',
                'clean_air_pace'
            ]
            
            print(f"🔧 Preparando {len(required_features)} características base:")
            for i, feature in enumerate(required_features, 1):
                print(f"   {i}. {feature}")
            
            feature_data = []
            feature_names = []
            
            # 1. Team encoding
            if 'team' in data.columns:
                team_encoded = self.label_encoder.fit_transform(data['team'].fillna('Unknown'))
                feature_data.append(team_encoded)
                feature_names.append('team_encoded')
                print(f"   ✅ team_encoded: {len(self.label_encoder.classes_)} equipos únicos")
            else:
                feature_data.append(np.zeros(len(data)))
                feature_names.append('team_encoded')
                print(f"   ⚠️ team_encoded: usando valores por defecto")
            
            # 2-4. Team historical performance (usando datos simulados por ahora)
            if 'team' in data.columns:
                # Crear performance histórico basado en el equipo
                team_perf_2024 = data['team'].map(self._get_team_performance_2024).fillna(10)
                team_perf_2023 = data['team'].map(self._get_team_performance_2023).fillna(10)
                team_perf_2022 = data['team'].map(self._get_team_performance_2022).fillna(10)
            else:
                team_perf_2024 = np.full(len(data), 10.0)
                team_perf_2023 = np.full(len(data), 10.0)
                team_perf_2022 = np.full(len(data), 10.0)
            
            feature_data.extend([team_perf_2024, team_perf_2023, team_perf_2022])
            feature_names.extend(['team_avg_position_2024', 'team_avg_position_2023', 'team_avg_position_2022'])
            print(f"   ✅ Performance histórico de equipos agregado")
            
            # 5. Grid position
            grid_pos = data.get('grid_position', data.get('quali_position', np.full(len(data), 10))).fillna(10)
            feature_data.append(grid_pos)
            feature_names.append('grid_position')
            print(f"   ✅ grid_position: rango {grid_pos.min():.1f} - {grid_pos.max():.1f}")
            
            # 6. Qualifying time
            quali_time = data.get('quali_best_time', data.get('q3_time', data.get('q2_time', data.get('q1_time', np.full(len(data), 90))))).fillna(90)
            feature_data.append(quali_time)
            feature_names.append('quali_best_time')
            print(f"   ✅ quali_best_time: rango {quali_time.min():.1f} - {quali_time.max():.1f}")
            
            # 7. Race best lap time
            race_time = data.get('race_best_lap_time', data.get('best_lap_time', np.full(len(data), 90))).fillna(90)
            feature_data.append(race_time)
            feature_names.append('race_best_lap_time')
            print(f"   ✅ race_best_lap_time: rango {race_time.min():.1f} - {race_time.max():.1f}")
            
            # 8. Clean air pace
            clean_pace = data.get('clean_air_pace', np.full(len(data), 0.0)).fillna(0.0)
            feature_data.append(clean_pace)
            feature_names.append('clean_air_pace')
            print(f"   ✅ clean_air_pace: rango {clean_pace.min():.3f} - {clean_pace.max():.3f}")
            
            # Convertir a matriz numpy
            X = np.column_stack(feature_data)
            
            print(f"✅ Características base preparadas: {X.shape}")
            return X, feature_names
            
        except Exception as e:
            print(f"❌ Error preparando características base: {e}")
            return None, None
    
    def _get_team_performance_2024(self, team):
        """Retorna la posición promedio del equipo en 2024"""
        team_performance = {
            'McLaren': 2.5,
            'Ferrari': 3.2,
            'Red Bull Racing': 1.8,
            'Mercedes': 4.1,
            'Aston Martin': 6.5,
            'Alpine': 8.3,
            'Williams': 9.1,
            'Racing Bulls': 8.7,
            'Haas': 9.8,
            'Sauber': 10.2
        }
        return team_performance.get(team, 10.0)
    
    def _get_team_performance_2023(self, team):
        """Retorna la posición promedio del equipo en 2023"""
        team_performance = {
            'Red Bull Racing': 1.2,
            'Mercedes': 4.5,
            'Ferrari': 5.1,
            'McLaren': 7.2,
            'Aston Martin': 3.8,
            'Alpine': 8.9,
            'Williams': 9.5,
            'Racing Bulls': 8.1,  # Era AlphaTauri
            'Haas': 9.8,
            'Sauber': 10.1  # Era Alfa Romeo
        }
        return team_performance.get(team, 10.0)
    
    def _get_team_performance_2022(self, team):
        """Retorna la posición promedio del equipo en 2022"""
        team_performance = {
            'Red Bull Racing': 2.1,
            'Ferrari': 3.5,
            'Mercedes': 4.8,
            'McLaren': 6.5,
            'Alpine': 7.2,
            'Racing Bulls': 8.8,  # Era AlphaTauri
            'Aston Martin': 7.9,
            'Williams': 9.1,
            'Haas': 8.5,
            'Sauber': 9.7  # Era Alfa Romeo
        }
        return team_performance.get(team, 10.0)
    
    def _add_additional_features(self, X_df, training_data):
        """Agrega características adicionales al dataset"""
        print("🔧 Agregando características adicionales...")
        
        additional_features = {}
        
        # Características de tiempo normalizadas
        for time_col in ['fp1_best_time', 'fp2_best_time', 'fp3_best_time']:
            if time_col in training_data.columns:
                normalized_col = f"{time_col}_normalized"
                if normalized_col in training_data.columns:
                    additional_features[normalized_col] = training_data[normalized_col].fillna(training_data[normalized_col].median())
        
        # Características de sector
        for sector_col in ['race_sector1', 'race_sector2', 'race_sector3']:
            if sector_col in training_data.columns:
                additional_features[sector_col] = training_data[sector_col].fillna(training_data[sector_col].median())
        
        # Características de práctica libre
        if 'fp_avg_position' in training_data.columns:
            additional_features['fp_avg_position'] = training_data['fp_avg_position'].fillna(10)
        
        if 'fp_consistency' in training_data.columns:
            additional_features['fp_consistency'] = training_data['fp_consistency'].fillna(0.5)
        
        # Características de rendimiento en carrera
        if 'race_pace_rank' in training_data.columns:
            additional_features['race_pace_rank'] = training_data['race_pace_rank'].fillna(10)
        
        if 'quali_performance' in training_data.columns:
            additional_features['quali_performance'] = training_data['quali_performance'].fillna(0.5)
        
        # Características de vuelta rápida normalizada
        if 'best_lap_time_normalized' in training_data.columns:
            additional_features['best_lap_time_normalized'] = training_data['best_lap_time_normalized'].fillna(training_data['best_lap_time_normalized'].median())
        
        # Crear DataFrame con características adicionales
        if additional_features:
            additional_df = pd.DataFrame(additional_features)
            X_df = pd.concat([X_df, additional_df], axis=1)
            print(f"   ✅ {len(additional_features)} características adicionales agregadas")
        else:
            print(f"   ⚠️ No se encontraron características adicionales válidas")
        
        return X_df
    
    def _clean_data(self, X_df, training_data, feature_names):
        """Limpia datos eliminando valores infinitos y NaN"""
        print("🧹 Limpiando datos...")
        
        # Reemplazar infinitos con NaN
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        
        # Mostrar estadísticas de valores faltantes
        missing_stats = X_df.isnull().sum()
        if missing_stats.sum() > 0:
            print(f"   📊 Valores faltantes encontrados:")
            for col, missing_count in missing_stats[missing_stats > 0].items():
                print(f"      {col}: {missing_count} ({missing_count/len(X_df)*100:.1f}%)")
        
        # Rellenar valores faltantes con la mediana
        for col in X_df.columns:
            if X_df[col].isnull().sum() > 0:
                median_val = X_df[col].median()
                if pd.isna(median_val):
                    # Si la mediana también es NaN, usar un valor por defecto
                    if 'position' in col.lower():
                        median_val = 10.0
                    elif 'time' in col.lower():
                        median_val = 90.0
                    else:
                        median_val = 0.0
                X_df[col].fillna(median_val, inplace=True)
        
        print(f"   ✅ Datos limpios: {X_df.shape}")
        return X_df, training_data
    
    def _prepare_target_variable(self, training_data):
        """Prepara la variable objetivo (final_position)"""
        print("🎯 Preparando variable objetivo...")
        
        # Buscar columna de posición final
        target_columns = ['final_position', 'race_position', 'position']
        target_col = None
        
        for col in target_columns:
            if col in training_data.columns:
                target_col = col
                break
        
        if target_col is None:
            print("❌ No se encontró variable objetivo válida")
            return None
        
        y = training_data[target_col].copy()
        
        # Limpiar variable objetivo
        y = y.fillna(20)  # Posición por defecto para valores faltantes
        y = np.clip(y, 1, 20)  # Asegurar que esté en rango válido
        
        print(f"   ✅ Variable objetivo preparada: {target_col}")
        print(f"   📊 Rango: {y.min()} - {y.max()}")
        print(f"   📊 Promedio: {y.mean():.2f}")
        
        return y
    
    def _filter_valid_data(self, X_df, y):
        """Filtra datos válidos eliminando filas con problemas"""
        print("🔍 Filtrando datos válidos...")
        
        initial_count = len(X_df)
        
        # Verificar que no haya NaN restantes
        mask_valid = ~(X_df.isnull().any(axis=1) | pd.isna(y))
        
        X_clean = X_df[mask_valid].copy()
        y_clean = y[mask_valid].copy()
        
        removed_count = initial_count - len(X_clean)
        if removed_count > 0:
            print(f"   ⚠️ {removed_count} filas eliminadas por datos inválidos")
        
        print(f"   ✅ Datos válidos: {len(X_clean)} muestras")
        
        return X_clean, y_clean