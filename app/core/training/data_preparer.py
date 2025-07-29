import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from app.core.features.feature_extractor import FeatureExtractor

class DataPreparer:
    """Prepara datos para entrenamiento de modelos ML"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor()
        self.feature_names = []
    
    def prepare_training_data(self, data):
        """Prepara datos completos para entrenamiento"""
        print("Preparando datos para entrenamiento...")
        
        if data is None or data.empty:
            print("No hay datos disponibles para preparar")
            return None, None, None, None, None
        
        training_data = data.copy()
        print(f"📊 Datos originales: {training_data.shape}")
        
        # 1. Preparar características base (5 obligatorias)
        X_base, feature_names = self._prepare_base_features(training_data)
        if X_base is None:
            return None, None, None, None, None
        
        X_df = pd.DataFrame(X_base, columns=feature_names)
        
        # 2. Agregar características adicionales
        X_df = self._add_additional_features(X_df, training_data)
        
        # 3. Limpiar datos
        X_clean, training_data_clean = self._clean_data(X_df, training_data, feature_names)
        
        # 4. Preparar variable objetivo
        y_clean = self._prepare_target_variable(training_data_clean)
        if y_clean is None:
            return None, None, None, None, None
        
        # 5. Filtrar datos válidos
        X_final, y_final = self._filter_valid_data(X_clean, y_clean)
        
        # 6. Dividir en entrenamiento y test
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42
        )
        
        self.feature_names = list(X_final.columns)
        
        print(f"✅ Datos preparados: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} test")
        print(f"📊 Total características: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test, self.label_encoder
    
    def _prepare_base_features(self, data):
        """Prepara las 5 características base obligatorias"""
        features_list = []
        feature_names = []
        
        # 1. Driver encoded
        if 'driver' not in data.columns:
            print("❌ Columna 'driver' no encontrada")
            return None, None
        
        driver_encoded = self.label_encoder.fit_transform(data['driver'])
        features_list.append(driver_encoded)
        feature_names.append('driver_encoded')
        
        # 2. Clean air pace
        clean_air_pace = self._get_clean_air_pace(data)
        features_list.append(clean_air_pace)
        feature_names.append('clean_air_pace')
        
        # 3-5. Sector times
        for i, sector_name in enumerate(['sector1_time', 'sector2_time', 'sector3_time'], 3):
            sector_time = self._get_sector_time(data, i-2, clean_air_pace)
            features_list.append(sector_time)
            feature_names.append(sector_name)
        
        return np.column_stack(features_list), feature_names
    
    def _get_clean_air_pace(self, data):
        """Obtiene clean air pace o usa proxy"""
        for col in ['clean_air_pace', 'race_best_lap_time', 'best_lap_time']:
            if col in data.columns:
                pace = pd.to_numeric(data[col], errors='coerce')
                return pace.fillna(pace.median() if not pace.isna().all() else 79.5)
        
        return pd.Series([79.5] * len(data))
    
    def _get_sector_time(self, data, sector_num, base_time):
        """Obtiene tiempo de sector o genera sintético"""
        for session in ['race', 'quali', 'fp3', 'fp2', 'fp1']:
            col = f'{session}_sector{sector_num}'
            if col in data.columns:
                sector_time = pd.to_numeric(data[col], errors='coerce')
                return sector_time.fillna(sector_time.median())
        
        # Genera un tiempo de sector estimado en base al clean_pace -> solo si no tiene los datos
        multipliers = {1: 0.30, 2: 0.40, 3: 0.30}
        return base_time * multipliers[sector_num]
    
    def _add_additional_features(self, X_df, training_data):
        """Agrega características adicionales de práctica/quali/carrera"""
        # Práctica libre
        practice_features = self.feature_extractor.extract_all_practice_features(training_data)
        if practice_features is not None and not practice_features.empty:
            for col in practice_features.columns:
                X_df[col] = practice_features[col]
        
        # Clasificación
        quali_features = self.feature_extractor.extract_all_qualifying_features(training_data)
        if quali_features is not None and not quali_features.empty:
            for col in quali_features.columns:
                X_df[col] = quali_features[col]
        
        # Carrera
        race_features = self.feature_extractor.extract_all_race_features(training_data)
        if race_features is not None and not race_features.empty:
            for col in race_features.columns:
                if col not in X_df.columns:
                    X_df[col] = race_features[col]
        
        # Derivadas
        derived_features = self.feature_extractor.extract_derived_features(training_data)
        if derived_features is not None and not derived_features.empty:
            for col in derived_features.columns:
                X_df[col] = derived_features[col]
        
        return X_df
    
    def _clean_data(self, X_df, training_data, feature_names):
        """Limpia y procesa los datos"""
        # Reemplazar infinitos con NaN
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        
        # Imputar valores faltantes
        numeric_columns = X_df.select_dtypes(include=[np.number]).columns
        X_df[numeric_columns] = X_df[numeric_columns].fillna(X_df[numeric_columns].median())
        
        # Filtrar filas válidas
        critical_mask = ~X_df[feature_names].isnull().all(axis=1)
        X_clean = X_df[critical_mask]
        training_data_clean = training_data[critical_mask]
        
        return X_clean, training_data_clean
    
    def _prepare_target_variable(self, data):
        """Prepara la variable objetivo"""
        target_options = [
            'clean_air_pace', 'race_best_lap_time', 'best_lap_time',
            'race_position', 'quali_position', 'grid_position'
        ]
        
        for col in target_options:
            if col in data.columns:
                y = pd.to_numeric(data[col], errors='coerce')
                if not y.isnull().all():
                    print(f"🎯 Variable objetivo: {col}")
                    return y
        
        print("❌ No se encontró variable objetivo válida")
        return None
    
    def _filter_valid_data(self, X, y):
        """Filtra datos válidos para entrenamiento"""
        valid_mask = ~y.isnull()
        X_final = X[valid_mask]
        y_final = y[valid_mask]
        
        if len(X_final) == 0:
            print("❌ No hay datos válidos después de filtrado")
            return None, None
        
        return X_final, y_final