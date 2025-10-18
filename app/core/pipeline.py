import pickle
import os
import pandas as pd
from datetime import datetime
from app.data.collectors.fastf1_collector import FastF1Collector
from app.data.preprocessors.data_cleaner import clean_data
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
from app.core.training.model_trainer import ModelTrainer
from app.core.utils.race_range_builder import RaceRangeBuilder
from app.core.predictors.simple_position_predictor import SimplePositionPredictor

# ===== NUEVOS IMPORTS PARA EXPORTACIÓN DE PIPELINE =====
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# ===== Fallback pickeable: alineador de columnas =====
class FeatureAligner(BaseEstimator, TransformerMixin):
    """
    Alinea columnas a feature_names.pkl si está disponible; si no, devuelve X tal cual.
    Es compatible con sklearn y 100% pickeable (sin lambdas).
    """
    def __init__(self, feature_names_path: str = "app/models_cache/feature_names.pkl"):
        self.feature_names_path = feature_names_path
        self.feature_names_ = None

    def fit(self, X, y=None):
        try:
            with open(self.feature_names_path, "rb") as f:
                names = pickle.load(f)
            if isinstance(names, (list, tuple)):
                self.feature_names_ = list(names)
        except Exception:
            self.feature_names_ = None
        return self

    def transform(self, X):
        import pandas as pd
        if self.feature_names_ is None:
            return X
        df = pd.DataFrame(X).copy()
        # Crear faltantes y reordenar
        for c in self.feature_names_:
            if c not in df.columns:
                df[c] = 0
        df = df.reindex(columns=self.feature_names_, fill_value=0)
        # Asegurar numérico
        for c in df.columns:
            if not np.issubdtype(df[c].dtype, np.number):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df


class Pipeline:
    """Pipeline principal con features avanzadas"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        
        # Componentes especializados con features avanzadas
        self.race_range_builder = RaceRangeBuilder()
        self.data_preparer = EnhancedDataPreparer()
        self.model_trainer = ModelTrainer()
        
        # Collector
        race_range = self.race_range_builder.build_race_range(config)
        self.collector = FastF1Collector(race_range)

    def run(self):
        
        """Ejecuta el pipeline completo con validación robusta"""
        print("🚀 Iniciando pipeline de entrenamiento...")
        
        # 1. Cargar o recolectar datos
        if not self._load_cached_data():
            print("📥 Recolectando datos frescos...")
            self.collect_data()
            self.preprocess_data()
            self._save_cached_data()

        
        
        # 2. Validar tamaño del dataset
        if self.data is None or len(self.data) < 30:
            data_len = len(self.data) if self.data is not None else 0
            print(f"⚠️  ADVERTENCIA: Dataset pequeño ({data_len} muestras)")
            print(f"   💡 Considera recolectar más datos para evitar overfitting")
        
        # 3. Guardar dataset original antes de entrenamiento
        self._save_dataset_before_training()
        
        # 4. Preparar datos para entrenamiento
        training_results = self.data_preparer.prepare_training_data(self.data)
        if training_results[0] is None:
            print("❌ Error preparando datos de entrenamiento")
            return False
        
        X_train, X_test, y_train, y_test, feature_names = training_results
        
        # 5. Guardar dataset después del feature engineering
        self._save_dataset_after_feature_engineering(X_train, X_test, y_train, y_test)

        # 6. Validar split de datos
        print(f"📊 Datos de entrenamiento: {len(X_train)} muestras")
        print(f"📊 Datos de test: {len(X_test)} muestras")
        
        if len(X_train) < 20:
            print(f"🚨 ADVERTENCIA: Muy pocos datos de entrenamiento")
            print(f"   💡 Cross-validation será limitado")
        
        # 7. Entrenar modelos con cross-validation (incluyendo pesos por años)
        model_trainer = ModelTrainer(use_time_series_cv=True)
        
        # Debug: verificar si los índices están disponibles
        train_indices = getattr(self.data_preparer, 'train_indices', None)
        print(f"🔍 DEBUG: train_indices en pipeline: {train_indices is not None}")
        if train_indices is not None:
            print(f"🔍 DEBUG: len(train_indices): {len(train_indices)}")
        
        results = model_trainer.train_all_models(
            X_train, X_test, y_train, y_test, 
            self.data_preparer.label_encoder, feature_names,
            df_original=self.data,  # Pasar datos originales para pesos por años
            train_indices=train_indices
        )
        
        # 8. Validar resultados
        successful_models = [name for name, metrics in results.items() if 'error' not in metrics]
        
        if not successful_models:
            print("❌ Ningún modelo se entrenó exitosamente")
            return False
        
        print(f"✅ Pipeline completado: {len(successful_models)} modelos entrenados")

        # 9. Exportar pipeline completo (preprocesamiento + modelo) sin lambdas (pickeable)
        try:
            # Preferimos un preprocesador real si tu DataPreparer lo expone; si no, fallback FeatureAligner
            preproc = getattr(self.data_preparer, 'preprocessor', None)
            if preproc is None and hasattr(self.data_preparer, 'get_inference_preprocessor'):
                try:
                    preproc = self.data_preparer.get_inference_preprocessor()
                except Exception:
                    preproc = None
            if preproc is None:
                preproc = FeatureAligner()

            # Cargar el mejor modelo guardado (orden de preferencia)
            model = None
            for p in [
                "app/models_cache/xgboost_model.pkl",
                "app/models_cache/randomforest_model.pkl",
                "app/models_cache/gradientboosting_model.pkl",
            ]:
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        model = pickle.load(f)
                    print(f"📦 Modelo cargado para exportación: {p}")
                    break
            if model is None:
                print("⚠️ No se encontró un modelo entrenado en models_cache. Omito exportación.")
            else:
                self.export_full_pipeline(model=model, feature_preprocessor=preproc)
        except Exception as e:
            print(f"⚠️ Exportación de pipeline omitida: {e}")

        return True

    def collect_data(self):
        """Recolecta datos de FastF1"""
        print("📡 Recolectando datos de FastF1...")
        self.collector.collect_data()
        self.data = self.collector.get_data()

    def preprocess_data(self):
        """Limpia y prepara los datos"""
        print("🧹 Limpiando datos...")
        self.data = clean_data(self.data)

    def predict_next_race_positions(self):
        """Predice posiciones para la próxima carrera"""
        print("🎯 Prediciendo posiciones para próxima carrera...")

        predictor = SimplePositionPredictor()
        predictions_df = predictor.predict_positions_2025()
        predictor.show_realistic_predictions(predictions_df)

        # Guardar predicciones
        output_file = "app/models_cache/realistic_predictions_2025.csv"
        predictions_df.to_csv(output_file, index=False)
        print(f"💾 Predicciones guardadas: {output_file}")

        return predictions_df

    def _load_cached_data(self):
        """Carga datos desde cache"""
        cache_file = "app/models_cache/cached_data.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print("📦 Datos cargados desde cache")
            return True
        return False
    
    def export_full_pipeline(self, model, feature_preprocessor):
        """
        Exporta un sklearn.Pipeline pickeable (sin lambdas), con etapas:
          - 'pre': feature_preprocessor (ColumnTransformer, otro preprocesador o FeatureAligner)
          - 'model': estimador final
        Guarda en app/models_cache/full_pipeline.pkl
        """
        try:
            full_pipeline = SkPipeline([
                ("pre", feature_preprocessor),
                ("model", model)
            ])
            path = "app/models_cache/full_pipeline.pkl"
            with open(path, "wb") as f:
                pickle.dump(full_pipeline, f)
            print(f"💾 Pipeline completo exportado en {path}")
        except Exception as e:
            print(f"❌ No se pudo exportar el pipeline completo: {e}")

    def _save_cached_data(self):
        """Guarda datos en cache"""
        cache_dir = "app/models_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "cached_data.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"💾 Datos guardados en cache: {cache_file}")

    def _save_dataset_before_training(self):
        """Guarda el dataset en CSV antes del entrenamiento"""
        if self.data is None or len(self.data) == 0:
            print("⚠️  No hay datos para guardar")
            return
        
        # Crear directorio si no existe
        cache_dir = "app/models_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Generar nombre de archivo con timestamp
        csv_file = os.path.join(cache_dir, f"dataset_before_training_latest.csv")
        
        try:
            # Convertir a DataFrame si no lo es ya
            if isinstance(self.data, pd.DataFrame):
                df = self.data.copy()
            else:
                df = pd.DataFrame(self.data)
            
            # Guardar como CSV
            df.to_csv(csv_file, index=False)
            print(f"📊 Dataset guardado antes del entrenamiento: {csv_file}")
            print(f"   📈 Forma del dataset: {df.shape}")
            print(f"   📋 Columnas: {list(df.columns)}")
            
            # También guardar una versión sin timestamp para referencia
            latest_file = os.path.join(cache_dir, "dataset_before_training_latest.csv")
            df.to_csv(latest_file, index=False)
            print(f"📊 Versión latest guardada: {latest_file}")
            
        except Exception as e:
            print(f"❌ Error guardando dataset: {e}")
            # Intentar guardar información básica
            try:
                info_file = os.path.join(cache_dir, f"dataset_info_latest.txt")
                with open(info_file, 'w') as f:
                    f.write(f"Dataset info:")
                    f.write(f"Type: {type(self.data)}")
                    f.write(f"Length: {len(self.data)}")
                    if hasattr(self.data, 'shape'):
                        f.write(f"Shape: {self.data.shape}")
                    if hasattr(self.data, 'columns'):
                        f.write(f"Columns: {list(self.data.columns)}")
                print(f"📝 Info del dataset guardada: {info_file}")
            except:
                print("❌ No se pudo guardar información del dataset")

    def _save_dataset_after_feature_engineering(self, X_train, X_test, y_train, y_test):
        """Guarda el dataset después del feature engineering y limpieza"""
        try:
            # Crear directorio si no existe
            cache_dir = "app/models_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Combinar datos de entrenamiento y test
            X_combined = pd.concat([X_train, X_test], ignore_index=True)
            y_combined = pd.concat([y_train, y_test], ignore_index=True)
            
            # Crear DataFrame final con features + target (sin columna 'split') y con 'year' si está disponible
            df_processed = X_combined.copy()
            # Intentar añadir 'year' como primera columna para legibilidad
            try:
                train_years = getattr(self.data_preparer, 'train_years', None)
                test_years = getattr(self.data_preparer, 'test_years', None)
                if train_years is not None and test_years is not None:
                    years_series = pd.concat([train_years, test_years], ignore_index=True)
                    if len(years_series) == len(df_processed):
                        df_processed.insert(0, 'year', years_series.reset_index(drop=True))
            except Exception:
                pass
            df_processed['target_position'] = y_combined
            
            # Guardar archivos
            csv_file = os.path.join(cache_dir, f"dataset_after_feature_engineering_latest.csv")
            df_processed.to_csv(csv_file, index=False)
            
            
            print(f"🧠 Dataset procesado guardado: {csv_file}")
            print(f"   📈 Forma final (features + target): {df_processed.shape}")
            print(f"   🎯 Features finales:({len(X_combined.columns)}) {list(X_combined.columns)}")
            print(f"   📊 Split: {len(X_train)} train + {len(X_test)} test = {len(df_processed)} total")
            
        except Exception as e:
            print(f"❌ Error guardando dataset procesado: {e}")
            try:
                # Información básica como fallback
                info_file = os.path.join(cache_dir, f"processed_dataset_info_latest.txt")
                with open(info_file, 'w') as f:
                    f.write(f"Processed Dataset Info:")
                    f.write(f"X_train shape: {X_train.shape}")
                    f.write(f"X_test shape: {X_test.shape}")
                    f.write(f"y_train length: {len(y_train)}")
                    f.write(f"y_test length: {len(y_test)}")
                    f.write(f"Features: {list(X_train.columns)}")
                print(f"📝 Info del dataset procesado guardada: {info_file}")
            except:
                print("❌ No se pudo guardar información del dataset procesado")
