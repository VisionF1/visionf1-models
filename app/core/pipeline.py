from app.data.collectors.fastf1_collector import FastF1Collector
from app.data.preprocessors.data_cleaner import clean_data
from app.core.features.driver_features import extract_driver_features
from app.core.features.lap_time_features import extract_best_lap_time
from app.core.features.sector_features import extract_sector_times  
from app.core.features.pace_features import extract_pace_features
from app.core.predictors.random_forest import RandomForestPredictor
from app.core.predictors.xgboost_model import XGBoostPredictor
from app.core.predictors.gradient_boosting import GradientBoostingPredictor
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.collector = FastF1Collector(self._build_race_range(config))
        self.models = {
            'RandomForest': RandomForestPredictor(),
            'XGBoost': XGBoostPredictor(),
            'GradientBoosting': GradientBoostingPredictor()
        }

    def _build_race_range(self, config):
        """Construye el rango de carreras basado en la configuración"""
        race_range = []
        year = config["start_year"]
        num_races = config["num_races"]
        
        # Para simplificar, asumimos números de carrera
        for i in range(1, num_races + 1):
            race_range.append({
                'year': year,
                'race_name': i  # Usar número de carrera
            })
        return race_range

    def run(self):
        """Ejecuta el pipeline completo"""
        if not self.load_cached_data():
            self.collect_data()
            self.preprocess_data()
            self.save_cached_data()
        
        self.extract_features()
        self.prepare_training_data()
        self.train_models()

    def collect_data(self):
        """Recolecta datos de FastF1"""
        print("Recolectando datos de FastF1...")
        self.collector.collect_data()
        self.data = self.collector.get_data()

    def preprocess_data(self):
        """Limpia y prepara los datos"""
        print("Limpiando datos...")
        self.data = clean_data(self.data)

    def save_cached_data(self):
        """Guarda los datos procesados en cache"""
        cache_dir = "app/models_cache"
        cache_file = os.path.join(cache_dir, "cached_data.pkl")
        
        # Crear directorio si no existe
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Datos guardados en cache: {cache_file}")

    def load_cached_data(self):
        """Carga datos desde cache si existen"""
        cache_file = "app/models_cache/cached_data.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print("Datos cargados desde cache.")
            return True
        return False

    def extract_features(self):
        """Extrae características de los datos"""
        print("Extrayendo características...")
        if self.data is None or self.data.empty:
            print("No hay datos para extraer características")
            return
            
        self.features = {
            'drivers': extract_driver_features(self.data),
            'best_lap_times': extract_best_lap_time(self.data),
            'sector_times': extract_sector_times(self.data),
            'pace_features': extract_pace_features(self.data)
        }

    def prepare_training_data(self):
        """Prepara los datos para entrenamiento en formato ML"""
        print("Preparando datos para entrenamiento...")
        
        if self.data is None or self.data.empty:
            print("No hay datos disponibles para preparar")
            return
        
        # Crear DataFrame con todas las características
        training_data = self.data.copy()
        
        # Eliminar filas con valores nulos en columnas críticas
        training_data = training_data.dropna(subset=['best_lap_time', 'clean_air_pace'])
        
        if training_data.empty:
            print("No hay datos válidos después de la limpieza")
            return
        
        # Preparar características (X)
        features_list = []
        
        # 1. Codificar driver como variable numérica
        driver_encoded = self.label_encoder.fit_transform(training_data['driver'])
        features_list.append(pd.DataFrame({'driver_encoded': driver_encoded}))
        
        # 2. Usar clean_air_pace como característica
        features_list.append(pd.DataFrame({'clean_air_pace': training_data['clean_air_pace'].values}))
        
        # 3. Extraer tiempos de sectores del diccionario
        sector_features = self._extract_sector_features_numeric(training_data['sector_times'])
        if sector_features is not None:
            features_list.append(sector_features)
        
        # Combinar todas las características
        X = pd.concat(features_list, axis=1)
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), 
            columns=X.columns,
            index=X.index
        )
        
        # Variable objetivo (y) - predecir best_lap_time
        y = training_data['best_lap_time'].values
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )
        
        print(f"Datos preparados: {self.X_train.shape[0]} samples para entrenamiento, {self.X_test.shape[0]} para prueba")
        print(f"Características: {list(X_imputed.columns)}")
        print(f"Valores NaN restantes: {self.X_train.isnull().sum().sum()}")

    def _extract_sector_features_numeric(self, sector_times_series):
        """Extrae características numéricas de los tiempos de sector"""
        try:
            sector_data = []
            for sector_dict in sector_times_series:
                if isinstance(sector_dict, dict):
                    # Extraer valores numéricos de cada sector
                    sector1 = sector_dict.get('Sector1Time', np.nan)
                    sector2 = sector_dict.get('Sector2Time', np.nan)  
                    sector3 = sector_dict.get('Sector3Time', np.nan)
                    
                    # Convertir a segundos si es necesario
                    if hasattr(sector1, 'total_seconds'):
                        sector1 = sector1.total_seconds()
                    if hasattr(sector2, 'total_seconds'):
                        sector2 = sector2.total_seconds()
                    if hasattr(sector3, 'total_seconds'):
                        sector3 = sector3.total_seconds()
                    
                    sector_data.append({
                        'sector1_time': sector1,
                        'sector2_time': sector2,
                        'sector3_time': sector3
                    })
                else:
                    sector_data.append({
                        'sector1_time': np.nan,
                        'sector2_time': np.nan,
                        'sector3_time': np.nan
                    })
            
            return pd.DataFrame(sector_data)
        except Exception as e:
            print(f"Error extrayendo características de sectores: {e}")
            return None

    def train_models(self):
        """Entrena todos los modelos y evalúa su rendimiento"""
        print("Entrenando modelos...")
        
        if self.X_train is None or self.y_train is None:
            print("No hay datos preparados para entrenar")
            return
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n--- Entrenando {name} ---")
            
            try:
                # Entrenar el modelo
                model.train(self.X_train, self.y_train)
                
                # Hacer predicciones
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calcular métricas
                train_mse = mean_squared_error(self.y_train, y_pred_train)
                test_mse = mean_squared_error(self.y_test, y_pred_test)
                train_mae = mean_absolute_error(self.y_train, y_pred_train)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                
                # Guardar resultados
                results[name] = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
                
                # Mostrar resultados
                print(f"MSE Entrenamiento: {train_mse:.4f}")
                print(f"MSE Prueba: {test_mse:.4f}")
                print(f"MAE Entrenamiento: {train_mae:.4f}")
                print(f"MAE Prueba: {test_mae:.4f}")
                print(f"R² Entrenamiento: {train_r2:.4f}")
                print(f"R² Prueba: {test_r2:.4f}")
                
                # Guardar modelo entrenado
                model_path = f"app/models_cache/{name.lower()}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Modelo guardado en: {model_path}")
                
            except Exception as e:
                print(f"Error entrenando {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Mostrar resumen comparativo
        self._show_model_comparison(results)
        
        return results

    def _show_model_comparison(self, results):
        """Muestra una comparación de los resultados de todos los modelos"""
        print("\n" + "="*60)
        print("RESUMEN COMPARATIVO DE MODELOS")
        print("="*60)
        
        for name, metrics in results.items():
            if 'error' not in metrics:
                print(f"\n{name}:")
                print(f"  Test MSE: {metrics['test_mse']:.4f}")
                print(f"  Test MAE: {metrics['test_mae']:.4f}")
                print(f"  Test R²:  {metrics['test_r2']:.4f}")
        
        # Encontrar el mejor modelo basado en MSE de prueba
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        if valid_models:
            best_model = min(valid_models.keys(), key=lambda x: valid_models[x]['test_mse'])
            print(f"\n🏆 Mejor modelo: {best_model} (MSE: {valid_models[best_model]['test_mse']:.4f})")

    def make_predictions(self):
        """Hace predicciones con modelos entrenados"""
        print("Cargando modelo entrenado para predicciones...")
    
        try:
            # Primero, asegurar que tenemos los datos preparados
            if self.X_test is None or self.y_test is None:
                print("Preparando datos para predicciones...")
                # Cargar datos desde cache si existen
                if not self.load_cached_data():
                    print("No hay datos en cache. Ejecuta 'python main.py d' primero")
                    return
                
                # Preparar los datos
                self.extract_features()
                self.prepare_training_data()
                
                if self.X_test is None:
                    print("No se pudieron preparar los datos de prueba")
                    return
        
            # Cargar el mejor modelo
            best_model_path = "app/models_cache/gradientboosting_model.pkl"
            with open(best_model_path, 'rb') as f:
                best_model = pickle.load(f)
        
            # Hacer predicciones
            predictions = best_model.predict(self.X_test)
            
            print(f"\n{'='*50}")
            print("PREDICCIONES DEL MODELO GRADIENTBOOSTING")
            print(f"{'='*50}")
            print(f"\nPredicciones vs Valores Reales:")
            print("-" * 50)
            
            total_error = 0
            for i, (pred, real) in enumerate(zip(predictions, self.y_test)):
                error = abs(pred - real)
                total_error += error
                print(f"Muestra {i+1:2d}: Predicho={pred:6.3f}s | Real={real:6.3f}s | Error={error:5.3f}s")
            
            # Estadísticas de rendimiento
            avg_error = total_error / len(predictions)
            mse = mean_squared_error(self.y_test, predictions)
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            
            print(f"\n{'='*50}")
            print("ESTADÍSTICAS DE RENDIMIENTO")
            print(f"{'='*50}")
            print(f"Error promedio:     {avg_error:.4f} segundos")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Mean Absolute Error:{mae:.4f}")
            print(f"R² Score:           {r2:.4f}")
            print(f"Total de muestras:  {len(predictions)}")
            
            # Mostrar información sobre los pilotos si está disponible
            if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                print(f"\n{'='*50}")
                print("INFORMACIÓN DE PILOTOS")
                print(f"{'='*50}")
                print("Pilotos en el dataset:")
                for i, driver in enumerate(self.label_encoder.classes_):
                    print(f"  {i+1:2d}. {driver}")
        
        except FileNotFoundError:
            print("❌ No se encontró modelo entrenado.")
            print("   Ejecuta 'python main.py e' primero para entrenar los modelos")
        except Exception as e:
            print(f"❌ Error en predicciones: {e}")
            import traceback
            traceback.print_exc()