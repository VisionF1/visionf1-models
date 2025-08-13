import pandas as pd
import numpy as np
import pickle
import os
from app.config import DRIVERS_2025, DATA_IMPORTANCE, PREDICTION_CONFIG, PENALTIES
from app.core.adapters.progressive_adapter import ProgressiveAdapter
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
import random

class SimplePositionPredictor:
    def __init__(self):
        self.drivers_config = DRIVERS_2025
        self.data_importance = DATA_IMPORTANCE
        self.penalties = PENALTIES
        self.progressive_adapter = ProgressiveAdapter()
        self.trained_model = None
        self.label_encoder = None
        self.scaler = None
        self.adjustments = {}  # Añadido para evitar el error de atributo
        self.enhanced_data_preparer = EnhancedDataPreparer(use_advanced_features=True)
        
    def load_best_model(self):
        """Carga el mejor modelo basado en métricas de rendimiento"""
        
        # 1️⃣ PRIMERO: Intentar cargar métricas de entrenamiento
        metrics_file = "app/models_cache/training_results.pkl"
        training_metrics = self._load_training_metrics(metrics_file)
        
        if training_metrics:
            best_model_name = self._select_best_model_from_metrics(training_metrics)
            if best_model_name:
                print(f"🎯 Mejor modelo detectado por métricas: {best_model_name}")
                return self._load_specific_model(best_model_name)
        
        # 2️⃣ FALLBACK: Usar orden de prioridad si no hay métricas
        print("⚠️  No se encontraron métricas, usando orden de prioridad...")
        return self._load_by_priority()

    def _load_training_metrics(self, metrics_file):
        """Carga métricas de entrenamiento guardadas por ModelTrainer"""
        try:
            if os.path.exists(metrics_file):
                with open(metrics_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"❌ Error cargando métricas: {e}")
        return None

    def _select_best_model_from_metrics(self, training_metrics):
        """Selecciona el mejor modelo basado en métricas de rendimiento"""
        print("🔍 Analizando métricas de modelos...")
        
        viable_models = []
        
        for model_name, metrics in training_metrics.items():
            if 'error' in metrics:
                print(f"   ❌ {model_name}: Error durante entrenamiento")
                continue
            
            cv_score = metrics.get('cv_mse_mean', float('inf'))
            overfitting_score = metrics.get('overfitting_score', float('inf'))
            test_r2 = metrics.get('test_r2', 0)
            
            # 🔥 CRITERIOS DE SELECCIÓN
            # 1. Sin overfitting severo (score < 2.0)
            # 2. Buen CV score (menor es mejor)
            # 3. Buen R² en test (mayor es mejor)
            
            if overfitting_score < 2.0 and test_r2 > 0.5:
                viable_models.append({
                    'name': model_name,
                    'cv_score': cv_score,
                    'overfitting': overfitting_score,
                    'test_r2': test_r2,
                    'overall_score': self._calculate_overall_score(cv_score, overfitting_score, test_r2)
                })
                
                print(f"   ✅ {model_name}: CV={cv_score:.4f}, Overfitting={overfitting_score:.2f}, R²={test_r2:.4f}")
            else:
                print(f"   🚫 {model_name}: Rechazado (Overfitting={overfitting_score:.2f}, R²={test_r2:.4f})")
        
        if not viable_models:
            print("   ⚠️  No hay modelos viables sin overfitting")
            return None
        
        # Seleccionar el mejor modelo
        best_model = min(viable_models, key=lambda x: x['overall_score'])
        
        print(f"\n🏆 MEJOR MODELO SELECCIONADO: {best_model['name']}")
        print(f"   📊 CV MSE: {best_model['cv_score']:.4f}")
        print(f"   🎯 Overfitting Score: {best_model['overfitting']:.2f}")
        print(f"   📈 Test R²: {best_model['test_r2']:.4f}")
        
        return best_model['name']

    def _calculate_overall_score(self, cv_score, overfitting_score, test_r2):
        """Calcula un score general para comparar modelos"""
        # Penalizar overfitting y CV score alto, premiar R² alto
        normalized_cv = cv_score / 100  # Normalizar CV score
        overfitting_penalty = max(0, overfitting_score - 1) * 2  # Penalizar overfitting > 1
        r2_bonus = (1 - test_r2) * 10  # Penalizar R² bajo
        
        return normalized_cv + overfitting_penalty + r2_bonus

    def _load_specific_model(self, model_name):
        """Carga un modelo específico por nombre"""
        model_mapping = {
            'RandomForest': 'randomforest_model.pkl',
            'XGBoost': 'xgboost_model.pkl', 
            'GradientBoosting': 'gradientboosting_model.pkl'
        }
        
        model_file = f"app/models_cache/{model_mapping.get(model_name, model_name.lower() + '_model.pkl')}"
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    self.trained_model = pickle.load(f)
                
                # Cargar encoder y metadata
                self._load_model_metadata()
                
                print(f"✅ Modelo {model_name} cargado exitosamente")
                return True
                
            except Exception as e:
                print(f"❌ Error cargando modelo {model_name}: {e}")
        
        return False

    def _load_by_priority(self):
        """Método fallback con orden de prioridad mejorado"""
        # Prioridad basada en estabilidad típica de modelos
        priority_models = [
            ("app/models_cache/randomforest_model.pkl", "RandomForest"),      # Más estable
            ("app/models_cache/xgboost_model.pkl", "XGBoost"),               # Balanceado
            ("app/models_cache/gradientboosting_model.pkl", "GradientBoosting") # Más propenso a overfitting
        ]
        
        for model_file, model_name in priority_models:
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        self.trained_model = pickle.load(f)
                    
                    self._load_model_metadata()
                    print(f"✅ Modelo cargado por prioridad: {model_name}")
                    return True
                    
                except Exception as e:
                    print(f"❌ Error cargando {model_name}: {e}")
                    continue
        
        print("❌ No se encontraron modelos válidos")
        return False

    def _load_model_metadata(self):
        """Carga encoder y scaler si existen"""
        try:
            # Cargar label encoder
            encoder_file = "app/models_cache/label_encoder.pkl"
            if os.path.exists(encoder_file):
                with open(encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Cargar scaler si existe
            scaler_file = "app/models_cache/feature_scaler.pkl"  
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
        except Exception as e:
            print(f"⚠️  Error cargando metadata: {e}")
    
    def predict_positions_2025(self, recent_data_2024=None):
        """Predice posiciones priorizando información de 2025"""
        print("🎯 Prediciendo posiciones PRIORIZANDO información 2025...")
        
        # Intentar cargar modelo entrenado
        model_loaded = self.load_best_model()
        
        predictions = []
        current_race_number = PREDICTION_CONFIG["next_race"].get("race_number", 1)
        
        for driver, config in self.drivers_config.items():
            try:
                # 1. Crear características para el modelo ML
                ml_features = self._create_ml_features(driver, config)
                
                # 2. Predecir con el modelo ML
                if model_loaded and self.trained_model is not None and ml_features is not None:
                    ml_prediction = self.trained_model.predict([ml_features])[0]
                    ml_position = self._convert_ml_to_position(ml_prediction)  # Sin tier hardcodeado
                else:
                    ml_position = None
                
                # 3. Calcular posición base de configuración 2025
                min_pos, max_pos = config["expected_range"]
                config_position = (min_pos + max_pos) / 2
                
                # 4. 🔥 USAR CONFIGURACIÓN SIMPLE DE IMPORTANCIA
                if ml_position is not None:
                    ml_weight = self.data_importance["ml_vs_config"]["ml_weight"]
                    config_weight = self.data_importance["ml_vs_config"]["config_weight"]
                    
                    predicted_position = (ml_position * ml_weight) + (config_position * config_weight)
                    prediction_source = f"🔥 ML({ml_weight*100:.0f}%)+Config({config_weight*100:.0f}%)"
                else:
                    predicted_position = config_position + random.uniform(-1.0, 1.0)
                    prediction_source = "🎯 Config2025Only"
                
                # 5. Aplicar penalizaciones simples
                if config.get("rookie", False):
                    predicted_position += self.penalties["rookie"]
                    driver_type = "🆕 Rookie"
                elif config.get("team_change", False):
                    predicted_position += self.penalties["team_change"]
                    driver_type = "🔄 Cambio equipo"
                else:
                    driver_type = "👤 Veterano"
                
                # 6. Limitar entre 1 y 20
                predicted_position = max(1, min(20, predicted_position))
                
                predictions.append({
                    'driver': driver,
                    'team': config["team"],
                    'predicted_position': predicted_position,
                    'ml_position': ml_position,
                    'config_position': config_position,
                    'driver_type': driver_type,
                    'prediction_source': prediction_source,
                    'expected_range': f"P{min_pos}-{max_pos}"
                })
                
            except Exception as e:
                print(f"⚠️  Error prediciendo {driver}: {e}")
                # Fallback simple
                min_pos, max_pos = config["expected_range"]
                base_position = (min_pos + max_pos) / 2 + random.uniform(-1.0, 1.0)
                
                predictions.append({
                    'driver': driver,
                    'team': config["team"],
                    'predicted_position': max(1, min(20, base_position)),
                    'ml_position': None,
                    'config_position': base_position,
                    'driver_type': "⚠️ Fallback",
                    'prediction_source': "📊 ConfigOnly",
                    'expected_range': f"P{min_pos}-{max_pos}"
                })
        
        # Crear DataFrame y aplicar adaptación progresiva
        df = pd.DataFrame(predictions)
        
        # APLICAR ADAPTACIÓN PROGRESIVA SI ESTÁ HABILITADA
        if self.penalties.get("use_progressive", False):
            df = self.progressive_adapter.apply_progressive_penalties(df, current_race_number)
        
        # Ordenar por posición predicha y asignar posiciones finales
        df = df.sort_values('predicted_position').reset_index(drop=True)
        df['final_position'] = range(1, len(df) + 1)
        
        # Calcular confianza
        df['confidence'] = df.apply(
            lambda x: max(70, 100 - abs(x['predicted_position'] - x['final_position']) * 8), 
            axis=1
        )
        
        return df
    
    def _create_ml_features(self, driver, config):
        """Crea características para el modelo ML usando el sistema avanzado de features"""
        try:
            # Crear datos base completos simulados para este piloto
            base_data = pd.DataFrame({
                'driver': [driver],
                'qualifying_position': [10],  # Posición media simulada
                'grid_position': [10],
                'team': [config["team"]],
                'session_type': ['Race'],
                'points_before_race': [0],
                'race_name': ['Hungarian Grand Prix'],  # Añadir race_name
                'season': [2025],
                'year': [2025],  # Añadir year también
                'round': [13],
                
                # Datos meteorológicos del escenario activo
                'session_air_temp': [self._get_weather_value('session_air_temp')],
                'session_track_temp': [self._get_weather_value('session_track_temp')],
                'session_humidity': [self._get_weather_value('session_humidity')],
                'session_rainfall': [self._get_weather_value('session_rainfall')],
                
                # Datos adicionales simulados
                'quali_best_time': [90.0],  # Tiempo simulado
                'race_best_lap_time': [92.0],
                'clean_air_pace': [self._estimate_driver_pace(driver, config)],
                'quali_gap_to_pole': [1.0],
                'fp1_gap_to_fastest': [0.5],
                
                # Más datos simulados para evitar errores
                'sector1_time': [30.0],
                'sector2_time': [35.0],
                'sector3_time': [25.0],
                'lap_time': [90.0],
                'position': [10],
                'fastest_lap': [False],
                'status': ['Finished']
            })
            
            # Usar el sistema avanzado de features
            result = self.enhanced_data_preparer.prepare_enhanced_features(base_data)
            
            # El método prepare_enhanced_features devuelve 4 elementos: X, y, label_encoder, feature_names
            if isinstance(result, tuple) and len(result) == 4:
                X, _, _, _ = result  # Solo necesitamos X para predicciones
            elif isinstance(result, tuple) and len(result) == 2:
                X, _ = result  # Formato alternativo (X, y)
            else:
                X = result     # Formato solo X
            
            if X is not None and len(X) > 0:
                return X.iloc[0].values  # Retornar la primera (y única) fila como array
            else:
                # Fallback a features básicas si falla el sistema avanzado
                return self._create_basic_ml_features(driver, config)
                
        except Exception as e:
            print(f"⚠️ Error creando features avanzadas para {driver}: {str(e)}")
            # Fallback a features básicas
            return self._create_basic_ml_features(driver, config)
    
    def _get_weather_value(self, weather_param):
        """Obtiene valores meteorológicos del escenario activo"""
        try:
            active_scenario = PREDICTION_CONFIG.get('active_scenario', 'default')
            scenarios = PREDICTION_CONFIG.get('weather_scenarios', {})
            
            if active_scenario in scenarios:
                scenario = scenarios[active_scenario]
                if weather_param in scenario:
                    return scenario[weather_param]
            
            # Valores por defecto
            defaults = {
                'session_air_temp': 25.0,
                'session_track_temp': 35.0,
                'session_humidity': 60.0,
                'session_rainfall': False
            }
            return defaults.get(weather_param, 0)
            
        except Exception:
            # Valores por defecto en caso de error
            defaults = {
                'session_air_temp': 25.0,
                'session_track_temp': 35.0,
                'session_humidity': 60.0,
                'session_rainfall': False
            }
            return defaults.get(weather_param, 0)
    
    def _create_basic_ml_features(self, driver, config):
        """Fallback: Crea características básicas si falla el sistema avanzado"""
        try:
            # 1. Driver encoded
            if self.label_encoder is not None:
                try:
                    driver_encoded = self.label_encoder.transform([driver])[0]
                except:
                    # Fallback si el label_encoder no funciona
                    if hasattr(self.label_encoder, 'classes_'):
                        driver_encoded = len(self.label_encoder.classes_) // 2
                    else:
                        driver_encoded = hash(driver) % 100
            else:
                driver_encoded = hash(driver) % 100
            
            # 2. Team encoded (nuevo)
            team_encoded = hash(config["team"]) % 20  # Simple encoding del equipo
            
            # 3-5. Team historical performance (REAL DATA)
            team_avg_2024 = self._get_team_real_performance(config["team"], 2024)
            team_avg_2023 = self._get_team_real_performance(config["team"], 2023) 
            team_avg_2022 = self._get_team_real_performance(config["team"], 2022)
            
            # 6. Clean air pace individual (estimado)
            clean_air_pace = self._estimate_driver_pace(driver, config)
            
            # 7-8. Sector times básicos
            sector1_time = clean_air_pace * 0.30
            sector2_time = clean_air_pace * 0.40
            
            # Crear vector con las nuevas características
            features = [
                driver_encoded,     # Feature 1: driver_encoded  
                team_encoded,       # Feature 2: team_encoded (NUEVO)
                team_avg_2024,      # Feature 3: team_avg_position_2024 (REAL)
                team_avg_2023,      # Feature 4: team_avg_position_2023 (REAL)
                team_avg_2022,      # Feature 5: team_avg_position_2022 (REAL)
                clean_air_pace,     # Feature 6: clean_air_pace
                sector1_time,       # Feature 7: sector1_time
                sector2_time        # Feature 8: sector2_time
            ]
                    
            # Aplicar scaler si existe
            if self.scaler is not None:
                features = self.scaler.transform([features])[0]
            
            return features
            
        except Exception as e:
            print(f"Error creando features básicas para {driver}: {e}")
            return None
    
    def _get_tier_features(self, tier):
        """Obtiene características típicas por tier de equipo"""
        # Características base por tier (estimadas para 2025)
        tier_data = {
            1: {  # McLaren (dominante)
                'best_lap_time': 78.5,
                'sector1_time': 24.2,
                'sector2_time': 31.8,
                'sector3_time': 22.5,
                'clean_air_pace': 79.2
            },
            2: {  # Ferrari, Red Bull, Mercedes
                'best_lap_time': 79.1,
                'sector1_time': 24.4,
                'sector2_time': 32.1,
                'sector3_time': 22.6,
                'clean_air_pace': 79.8
            },
            3: {  # Williams, Racing Bulls
                'best_lap_time': 79.8,
                'sector1_time': 24.7,
                'sector2_time': 32.4,
                'sector3_time': 22.7,
                'clean_air_pace': 80.5
            },
            4: {  # Aston Martin, Haas
                'best_lap_time': 80.3,
                'sector1_time': 24.9,
                'sector2_time': 32.7,
                'sector3_time': 22.7,
                'clean_air_pace': 81.1
            },
            5: {  # Alpine, Sauber
                'best_lap_time': 80.8,
                'sector1_time': 25.1,
                'sector2_time': 33.0,
                'sector3_time': 22.7,
                'clean_air_pace': 81.6
            }
        }
        
        # Agregar algo de variabilidad realista
        base_data = tier_data.get(tier, tier_data[3])
        
        return {
            key: value + random.uniform(-0.1, 0.1) 
            for key, value in base_data.items()
        }
    
    def _get_team_real_performance(self, team, year):
        """Obtiene el rendimiento histórico REAL del equipo para un año específico"""
        try:
            import os
            import pickle
            import pandas as pd
            from app.core.utils.team_mapping_utils import quick_team_mapping
            
            cache_dir = "app/models_cache/raw_data"
            team_positions = []
            
            # Buscar archivos de datos del año específico
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    if str(year) in filename and filename.endswith('.pkl'):
                        try:
                            filepath = os.path.join(cache_dir, filename)
                            with open(filepath, 'rb') as f:
                                race_data = pickle.load(f)
                            
                            # Convertir a DataFrame y aplicar mapeo de equipos
                            temp_data = []
                            for driver, driver_data in race_data.items():
                                if 'team' in driver_data and 'race_position' in driver_data:
                                    temp_data.append({
                                        'driver': driver,
                                        'team': driver_data['team'],
                                        'race_position': driver_data['race_position']
                                    })
                            
                            if temp_data:
                                temp_df = pd.DataFrame(temp_data)
                                # 🔧 APLICAR MAPEO DE EQUIPOS A DATOS HISTÓRICOS
                                temp_df = quick_team_mapping(temp_df)
                                
                                # Extraer posiciones del equipo (ya mapeado)
                                team_data = temp_df[temp_df['team'] == team]
                                for _, row in team_data.iterrows():
                                    position = row['race_position']
                                    if isinstance(position, (int, float)) and not pd.isna(position):
                                        team_positions.append(position)
                                        
                        except Exception:
                            continue
            
            # Retornar promedio o valor por defecto
            if team_positions:
                return np.mean(team_positions)
            else:
                # Fallback basado en conocimiento general del equipo
                team_defaults = {
                    'McLaren': 6.5, 'Ferrari': 7.0, 'Red Bull Racing': 4.5,
                    'Mercedes': 8.0, 'Alpine': 12.0, 'Aston Martin': 11.0,
                    'Williams': 13.5, 'Racing Bulls': 14.0, 'Haas': 15.0, 'Sauber': 16.5
                }
                return team_defaults.get(team, 10.5)
                
        except Exception as e:
            return 10.5  # Posición media por defecto
    
    def _estimate_driver_pace(self, driver, config):
        """Estima el pace individual del piloto SIN usar tiers"""
        # Base pace por equipo (datos aproximados realistas)
        team_base_pace = {
            'McLaren': 79.0,
            'Ferrari': 79.3,
            'Red Bull Racing': 79.2,
            'Mercedes': 79.7,
            'Williams': 80.2,
            'Racing Bulls': 80.4,
            'Aston Martin': 80.6,
            'Haas': 80.8,
            'Alpine': 81.1,
            'Sauber': 81.4
        }
        
        base_pace = team_base_pace.get(config.get("team", "McLaren"), 80.0)
        
        # Ajustes individuales conocidos (datos reales)
        driver_adjustments = {
            'VER': -0.3, 'HAM': -0.2, 'LEC': -0.2, 'NOR': -0.1,
            'RUS': 0.0, 'PIA': 0.0, 'ALO': -0.1, 'SAI': 0.1,
            'ALB': 0.2, 'GAS': 0.1, 'OCO': 0.2, 'HUL': 0.3,
            'TSU': 0.4, 'STR': 0.3, 'ANT': 0.5, 'BEA': 0.6,
            'LAW': 0.7, 'HAD': 0.6, 'COL': 0.4, 'BOR': 0.8
        }
        
        adjustment = driver_adjustments.get(driver, 0.0)
        return base_pace + adjustment
    
    def _convert_ml_to_position(self, ml_prediction, tier=None):
        """Convierte la predicción ML a posición estimada SIN ajustes hardcodeados"""
        # El modelo ahora debería predecir posiciones directamente
        # O si predice tiempo, convertir de manera más neutral
        
        if isinstance(ml_prediction, (int, float)):
            # Si el modelo predice tiempo de vuelta
            if ml_prediction > 70:  # Parece tiempo de vuelta
                if ml_prediction < 79:
                    base_position = random.uniform(1, 4)
                elif ml_prediction < 80:
                    base_position = random.uniform(3, 8)
                elif ml_prediction < 81:
                    base_position = random.uniform(6, 12)
                elif ml_prediction < 82:
                    base_position = random.uniform(10, 16)
                else:
                    base_position = random.uniform(14, 20)
            else:
                # Si el modelo predice posición directamente
                base_position = max(1, min(20, ml_prediction))
        else:
            base_position = 10  # Default
        
        # YA NO usar tier adjustments - dejar que el modelo aprenda naturalmente
        # El modelo ahora tiene información real del equipo vía las features
        
        return max(1, min(20, base_position))
    
    def _predict_basic_method(self, current_race_number):
        """Método básico cuando no hay modelo ML disponible"""
        predictions = []
        
        for driver, config in self.drivers_config.items():
            min_pos, max_pos = config["expected_range"]
            base_position = (min_pos + max_pos) / 2
            predicted_position = base_position + random.uniform(-1.5, 1.5)
            
            if config.get("rookie", False):
                driver_type = "🆕 Rookie"
            elif config.get("team_change", False):
                driver_type = "🔄 Cambio equipo"
            else:
                driver_type = "👤 Veterano"
            
            predictions.append({
                'driver': driver,
                'team': config["team"],
                'predicted_position': max(1, min(20, predicted_position)),
                'ml_position': None,
                'config_position': base_position,
                'driver_type': driver_type,
                'prediction_source': "📊 Básico",
                'expected_range': f"P{min_pos}-{max_pos}"
            })
        
        df = pd.DataFrame(predictions)
        
        if self.adjustments.get("use_progressive_adaptation", False):
            df = self.progressive_adapter.apply_progressive_penalties(df, current_race_number)
        
        df = df.sort_values('predicted_position').reset_index(drop=True)
        df['final_position'] = range(1, len(df) + 1)
        df['confidence'] = df.apply(
            lambda x: max(70, 100 - abs(x['predicted_position'] - x['final_position']) * 8), 
            axis=1
        )
        
        return df

    def show_realistic_predictions(self, predictions_df):
        """Muestra predicciones de manera realista organizadas"""
        current_race_name = PREDICTION_CONFIG["next_race"].get("race_name", "Carrera Desconocida")

        
        print(f"\n{'='*100}")
        print(f"🏆 PREDICCIONES ML + CONFIGURACIÓN 2025 - CARRERA #{current_race_name}")
        print(f"{'='*100}")
        
        print(f"{'Pos':<4} {'Piloto':<6} {'Equipo':<16} {'Tipo':<15} {'Fuente':<12} {'Conf.'}")
        print("-" * 100)
        
        for _, row in predictions_df.iterrows():
            source = row.get('prediction_source', '📊 Config')
            
            print(f"P{row['final_position']:<3} {row['driver']:<6} {row['team']:<16} "
                  f"{row['driver_type']:<15} {source:<12} {row['confidence']:.0f}%")
        
        # Mostrar estadísticas de fuentes de predicción
        if 'prediction_source' in predictions_df.columns:
            source_stats = predictions_df['prediction_source'].value_counts()
            print(f"\n📊 Fuentes de predicción:")
            for source, count in source_stats.items():
                print(f"   {source}: {count} pilotos")
        
        # Análisis por equipos
        print(f"\n{'='*100}")
        print("📊 ANÁLISIS POR EQUIPOS")
        print(f"{'='*100}")
        
        team_analysis = predictions_df.groupby('team').agg({
            'final_position': ['min', 'mean'],
            'confidence': 'mean'
        }).round(1)
        
        team_analysis.columns = ['Mejor_Pos', 'Pos_Promedio', 'Confianza_Avg']
        team_analysis = team_analysis.sort_values('Mejor_Pos')
        
        for team, data in team_analysis.iterrows():
            print(f"{team:<16}: Mejor P{data['Mejor_Pos']:<2.0f} | "
                  f"Promedio P{data['Pos_Promedio']:<4.1f} | "
                  f"Confianza {data['Confianza_Avg']:.0f}%")
        
        # Top 3 más realistas
        print(f"\n🎯 Predicciones más confiables:")
        top_3 = predictions_df.nlargest(3, 'confidence')
        for _, row in top_3.iterrows():
            source = row.get('prediction_source', '📊 Config')
            print(f"  P{row['final_position']} - {row['driver']} ({row['team']}) - {row['confidence']:.0f}% - {source}")
        
        # Mostrar pilotos en proceso de adaptación
        if 'adaptation_penalty' in predictions_df.columns:
            adaptation_mask = (predictions_df['adaptation_penalty'].notna()) & (predictions_df['adaptation_penalty'] > 0)
            adapting_drivers = predictions_df[adaptation_mask]
            
            if not adapting_drivers.empty:
                print(f"\n⏳ Pilotos aún adaptándose (carrera #{current_race_name}):")
                for _, driver in adapting_drivers.iterrows():
                    penalty = driver.get('adaptation_penalty', 0)
                    progress = driver.get('adaptation_progress', 0)
                    print(f"   {driver['driver']} ({driver['team']}): {progress}% adaptado, "
                          f"penalización actual: +{penalty:.1f} posiciones")
            else:
                print(f"\n✅ Todos los pilotos ya están completamente adaptados (carrera #{current_race_name})")
        else:
            print(f"\n✅ Todos los pilotos ya están completamente adaptados (carrera #{current_race_name})")