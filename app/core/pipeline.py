import pickle
import os
from app.data.collectors.fastf1_collector import FastF1Collector
from app.data.preprocessors.data_cleaner import clean_data
from app.core.training.data_preparer import DataPreparer
from app.core.training.model_trainer import ModelTrainer
from app.core.utils.race_range_builder import RaceRangeBuilder
from app.core.predictors.simple_position_predictor import SimplePositionPredictor

class Pipeline:
    """Pipeline principal"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        
        # Componentes especializados
        self.race_range_builder = RaceRangeBuilder()
        self.data_preparer = DataPreparer()
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
        
        # 3. Preparar datos para entrenamiento
        training_results = self.data_preparer.prepare_training_data(self.data)
        if training_results[0] is None:
            print("❌ Error preparando datos de entrenamiento")
            return False
        
        X_train, X_test, y_train, y_test, label_encoder = training_results
        
        # 4. Validar split de datos
        print(f"📊 Datos de entrenamiento: {len(X_train)} muestras")
        print(f"📊 Datos de test: {len(X_test)} muestras")
        
        if len(X_train) < 20:
            print(f"🚨 ADVERTENCIA: Muy pocos datos de entrenamiento")
            print(f"   💡 Cross-validation será limitado")
        
        # 5. Entrenar modelos con cross-validation
        model_trainer = ModelTrainer(use_time_series_cv=True)
        results = model_trainer.train_all_models(
            X_train, X_test, y_train, y_test, 
            label_encoder, self.data_preparer.feature_names
        )
        
        # 6. Validar resultados
        successful_models = [name for name, metrics in results.items() if 'error' not in metrics]
        
        if not successful_models:
            print("❌ Ningún modelo se entrenó exitosamente")
            return False
        
        print(f"✅ Pipeline completado: {len(successful_models)} modelos entrenados")
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

    def _save_cached_data(self):
        """Guarda datos en cache"""
        cache_dir = "app/models_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "cached_data.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"💾 Datos guardados en cache: {cache_file}")