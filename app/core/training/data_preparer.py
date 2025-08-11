import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression  # Para selección de features
from app.core.features.feature_extractor import FeatureExtractor
from app.core.utils.robust_imputer import RobustF1Imputer  # NUEVO: Imputer robusto

class DataPreparer:
    """Prepara datos para entrenamiento de modelos ML con estrategias anti-overfitting"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor()
        self.feature_names = []
        self.imputer = RobustF1Imputer()  # NUEVO: Imputer especializado para F1
        self.feature_selector = None  # NUEVO: Selector de features
    
    def prepare_training_data(self, data):
        """Prepara datos completos para entrenamiento con estrategias anti-overfitting"""
        print("🔧 Preparando datos para entrenamiento (ANTI-OVERFITTING)...")
        
        if data is None or data.empty:
            print("❌ No hay datos disponibles para preparar")
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
        
        # 3. NUEVO: Limpiar datos con imputer robusto
        X_clean, training_data_clean = self._clean_data_robust(X_df, training_data, feature_names)
        
        # 4. Preparar variable objetivo
        y_clean = self._prepare_target_variable(training_data_clean)
        if y_clean is None:
            return None, None, None, None, None

        # 5. Filtrar datos válidos
        X_final, y_final = self._filter_valid_data(X_clean, y_clean)
        
        # 6. NUEVO: Selección de features para evitar overfitting
        X_selected = self._select_best_features(X_final, y_final)
        
        # 7. NUEVO: Validación de distribución para estratificación
        X_train, X_test, y_train, y_test = self._robust_train_test_split(X_selected, y_final)

        if X_final is not None:
            self.feature_names = list(X_selected.columns)
        else:
            self.feature_names = []

        print(f"✅ Datos preparados: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} test")
        print(f"📊 Features seleccionadas: {len(self.feature_names)} de {X_final.shape[1]} originales")
        
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
    
    def _clean_data_robust(self, X_df, training_data, feature_names):
        """Limpia datos con estrategias robustas anti-overfitting"""
        print("🧹 Limpieza robusta de datos...")
        
        # 1. Reemplazar infinitos con NaN
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Detectar outliers extremos (más de 3 desviaciones estándar)
        numeric_columns = X_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_df[col].std() > 0:  # Solo si hay variación
                z_scores = np.abs((X_df[col] - X_df[col].mean()) / X_df[col].std())
                outlier_mask = z_scores > 3
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    print(f"⚠️  Outliers detectados en {col}: {outlier_count}")
                    # Reemplazar outliers con percentiles 5 y 95
                    p5, p95 = X_df[col].quantile([0.05, 0.95])
                    X_df.loc[X_df[col] < p5, col] = p5
                    X_df.loc[X_df[col] > p95, col] = p95
        
        # 3. Imputación robusta con imputer especializado
        try:
            print("🔧 Aplicando imputación robusta para F1...")
            X_df_clean = self.imputer.fit_transform(X_df)
            
            # Verificar que no queden NaN
            nan_counts = X_df_clean.isnull().sum()
            total_nans = nan_counts.sum()
            
            if total_nans > 0:
                print(f"⚠️  {total_nans} NaN restantes después de imputación")
                # Fallback: rellenar con 0
                X_df_clean = X_df_clean.fillna(0)
                print("✅ NaN restantes rellenados con 0")
            else:
                print("✅ Todos los NaN exitosamente imputados")
                
        except Exception as e:
            print(f"⚠️  Error en imputación robusta: {e}")
            print("🔄 Fallback a imputación simple...")
            # Fallback a método anterior
            numeric_columns = X_df.select_dtypes(include=[np.number]).columns
            X_df_clean = X_df.copy()
            X_df_clean[numeric_columns] = X_df_clean[numeric_columns].fillna(X_df_clean[numeric_columns].median())
            X_df_clean = X_df_clean.fillna(0)
        
        # 4. Filtrar filas válidas (más conservador)
        # Al menos 70% de features deben ser válidas
        min_valid_features = max(3, int(0.7 * len(feature_names)))
        valid_features_count = (~X_df_clean[feature_names].isnull()).sum(axis=1)
        critical_mask = valid_features_count >= min_valid_features
        
        print(f"📊 Filas válidas: {critical_mask.sum()} de {len(critical_mask)} ({critical_mask.mean():.1%})")
        
        # Asegurar alineación por posición
        X_clean = X_df_clean[critical_mask].reset_index(drop=True)
        training_data_clean = training_data.reset_index(drop=True).iloc[critical_mask.values].reset_index(drop=True)
        
        return X_clean, training_data_clean
    
    def _select_best_features(self, X_final, y_final):
        """Selecciona las mejores features para evitar overfitting"""
        if X_final is None or len(X_final.columns) <= 10:
            print("📊 Dataset pequeño, manteniendo todas las features")
            return X_final
        
        print("🎯 Seleccionando mejores features...")
        
        # Número máximo de features basado en tamaño de dataset
        n_samples = len(X_final)
        max_features = min(
            len(X_final.columns),
            max(10, n_samples // 20)  # Máximo 1 feature por cada 20 samples
        )
        
        try:
            # Usar SelectKBest con f_regression
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected_array = selector.fit_transform(X_final, y_final)
            
            # Obtener nombres de features seleccionadas
            selected_features = X_final.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected_array, columns=selected_features, index=X_final.index)
            
            print(f"✅ Features seleccionadas: {len(selected_features)} de {len(X_final.columns)}")
            print(f"📊 Top 5 features: {selected_features[:5]}")
            
            self.feature_selector = selector
            return X_selected
            
        except Exception as e:
            print(f"⚠️  Error en selección de features: {e}")
            print("📊 Manteniendo todas las features")
            return X_final
    
    def _robust_train_test_split(self, X, y):
        """División robusta de datos con mejor estratificación"""
        # CRÍTICO: Verificar y eliminar cualquier NaN restante
        print("🔍 Verificación final de NaN antes de división...")
        
        # Verificar X
        X_nan_count = X.isnull().sum().sum()
        if X_nan_count > 0:
            print(f"⚠️  {X_nan_count} NaN encontrados en X, eliminando...")
            X = X.fillna(0)
            
        # Verificar y
        y_nan_count = y.isnull().sum()
        if y_nan_count > 0:
            print(f"⚠️  {y_nan_count} NaN encontrados en y, eliminando filas...")
            valid_y_mask = ~y.isnull()
            X = X[valid_y_mask]
            y = y[valid_y_mask]
        
        # Verificar infinitos
        inf_mask = np.isinf(X.select_dtypes(include=[np.number])).any(axis=1)
        if inf_mask.any():
            print(f"⚠️  {inf_mask.sum()} filas con infinitos, eliminando...")
            X = X[~inf_mask]
            y = y[~inf_mask]
        
        print(f"✅ Datos finales limpios: {len(X)} muestras")
        
        try:
            # Para regresión, crear bins para estratificación
            n_bins = min(5, len(y) // 20)  # Al menos 20 samples por bin
            
            if n_bins >= 2:
                y_binned = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=y_binned
                )
                print("✅ División estratificada aplicada")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                print("✅ División simple aplicada")
            
            # Verificación final post-split
            self._final_data_validation(X_train, X_test, y_train, y_test)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"⚠️  Error en división estratificada: {e}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self._final_data_validation(X_train, X_test, y_train, y_test)
            return X_train, X_test, y_train, y_test
    
    def _final_data_validation(self, X_train, X_test, y_train, y_test):
        """Validación final antes del entrenamiento"""
        print("🔍 Validación final de datos...")
        
        # Verificar NaN
        for name, data in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
            if hasattr(data, 'isnull'):
                nan_count = data.isnull().sum()
                if hasattr(nan_count, 'sum'):
                    nan_count = nan_count.sum()
                if nan_count > 0:
                    print(f"❌ ERROR: {name} contiene {nan_count} NaN")
                else:
                    print(f"✅ {name}: Sin NaN")
        
        # Verificar infinitos
        for name, data in [("X_train", X_train), ("X_test", X_test)]:
            if hasattr(data, 'select_dtypes'):
                inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
                if inf_count > 0:
                    print(f"❌ ERROR: {name} contiene {inf_count} infinitos")
                else:
                    print(f"✅ {name}: Sin infinitos")
        
        # Verificar shapes
        print(f"📊 Shapes: X_train{X_train.shape}, X_test{X_test.shape}, y_train{len(y_train)}, y_test{len(y_test)}")
        
        return True

    
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