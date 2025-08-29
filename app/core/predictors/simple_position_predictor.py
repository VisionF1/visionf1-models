import pandas as pd
import numpy as np
import pickle
import os
from app.config import DRIVERS_2025, PREDICTION_CONFIG, PENALTIES
from app.core.adapters.progressive_adapter import ProgressiveAdapter
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
import random

class SimplePositionPredictor:
    def __init__(self):
        self.drivers_config = DRIVERS_2025
        self.penalties = PENALTIES
        self.progressive_adapter = ProgressiveAdapter()
        self.trained_model = None
        self.label_encoder = None
        self.scaler = None
        self.adjustments = {}
        # Modo silencioso para evitar logs repetidos por piloto en predicción
        self.enhanced_data_preparer = EnhancedDataPreparer(quiet=True)
        
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
                    predicted_position = self._convert_ml_to_position(ml_prediction)  # Sin tier hardcodeado
                    prediction_source = "🤖 ML100%"
                else:
                    # Fallback solo en caso de error del modelo
                    predicted_position = 10.0 + random.uniform(-3.0, 3.0)  # Posición media con variación
                    prediction_source = "� Fallback"
                
                # 3. Penalización y etapa: rookies (penalización fija) vs cambios de equipo (progresiva)
                if config.get("rookie", False):
                    # Rookie: penalización fija por experiencia; sin etapas de adaptación
                    predicted_position += self.penalties.get("rookie", 2.5)
                    driver_type = "🆕 Rookie"
                elif config.get("team_change", False):
                    # Cambio de equipo: solo mostrar etapa (la penalización se aplica luego vía adaptador progresivo)
                    adaptation = self.progressive_adapter.get_adaptation_status(driver, current_race_number)
                    prefix = "🔄 Cambio equipo"
                    if adaptation.get("status") == "fully_adapted":
                        driver_type = f"{prefix} ✓ Adaptado"
                    else:
                        prog = int(adaptation.get("progress", 0))
                        stage = "Reciente" if prog < 25 else "En adaptación"
                        driver_type = f"{prefix} · {stage} ({prog}%)"
                else:
                    driver_type = "👤 Veterano"
                
                # 4. Limitar entre 1 y 20
                predicted_position = max(1, min(20, predicted_position))
                
                predictions.append({
                    'driver': driver,
                    'team': config["team"],
                    'predicted_position': predicted_position,
                    'driver_type': driver_type,
                    'prediction_source': prediction_source
                })
                
            except Exception as e:
                print(f"⚠️  Error prediciendo {driver}: {e}")
                # Fallback simple sin configuración
                base_position = 10.0 + random.uniform(-5.0, 5.0)  # Posición media con variación amplia
                
                predictions.append({
                    'driver': driver,
                    'team': config["team"],
                    'predicted_position': max(1, min(20, base_position)),
                    'driver_type': "⚠️ Error",
                    'prediction_source': "� Fallback"
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
            # Obtener carrera actual de la configuración
            current_race_name = PREDICTION_CONFIG["next_race"].get("race_name", "Hungarian Grand Prix")
            current_race_number = PREDICTION_CONFIG["next_race"].get("race_number", 13)
            
            # Generar datos realistas específicos por piloto
            driver_pace = self._estimate_driver_pace(driver, config)
            
            # Datos específicos por piloto basados en rendimiento y equipo
            team_performance = self._get_team_performance_estimate(config["team"])
            driver_performance = self._get_driver_performance_estimate(driver)
            
            # Crear datos base completos simulados para este piloto
            base_data = pd.DataFrame({
                'driver': [driver],
                'qualifying_position': [driver_performance.get('avg_quali', 10)],  
                'grid_position': [driver_performance.get('avg_grid', 10)],
                'team': [config["team"]],
                'session_type': ['Race'],
                'points_before_race': [driver_performance.get('season_points', 0)],
                'race_name': [current_race_name],  # Usar carrera actual
                'season': [2025],
                'year': [2025],  # Añadir year también
                'round': [current_race_number],
                
                # Datos meteorológicos del escenario activo
                'session_air_temp': [self._get_weather_value('session_air_temp')],
                'session_track_temp': [self._get_weather_value('session_track_temp')],
                'session_humidity': [self._get_weather_value('session_humidity')],
                'session_rainfall': [self._get_weather_value('session_rainfall')],
                
                # Datos específicos por piloto
                'quali_best_time': [driver_pace + 0.5],  # Quali ligeramente más rápido
                'race_best_lap_time': [driver_pace + 1.0],  # Race más lento
                'clean_air_pace': [driver_pace],
                'quali_gap_to_pole': [driver_performance.get('gap_to_pole', 1.0)],
                'fp1_gap_to_fastest': [driver_performance.get('fp1_gap', 0.5)],
                
                # Datos específicos por circuito y piloto
                'sector1_time': [driver_pace * 0.32],  # ~32% del tiempo total
                'sector2_time': [driver_pace * 0.38],  # ~38% del tiempo total
                'sector3_time': [driver_pace * 0.30],  # ~30% del tiempo total
                'lap_time': [driver_pace],
                'position': [driver_performance.get('avg_position', 10)],
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
    
    def _get_team_performance_estimate(self, team):
        """Estima rendimiento del equipo basado en datos reales 2025"""
        team_stats = {
            'McLaren': {'avg_position': 4.5, 'competitiveness': 9.0},
            'Ferrari': {'avg_position': 5.2, 'competitiveness': 8.5},
            'Red Bull Racing': {'avg_position': 6.0, 'competitiveness': 8.0},
            'Mercedes': {'avg_position': 6.8, 'competitiveness': 7.5},
            'Aston Martin': {'avg_position': 8.5, 'competitiveness': 6.5},
            'Williams': {'avg_position': 9.2, 'competitiveness': 6.0},
            'Alpine': {'avg_position': 10.5, 'competitiveness': 5.5},
            'Haas F1 Team': {'avg_position': 12.0, 'competitiveness': 5.0},
            'Racing Bulls': {'avg_position': 13.5, 'competitiveness': 4.5},
            'Kick Sauber': {'avg_position': 15.0, 'competitiveness': 4.0},
        }
        return team_stats.get(team, {'avg_position': 10.0, 'competitiveness': 5.0})
    
    def _get_driver_performance_estimate(self, driver):
        """Estima rendimiento individual del piloto basado en temporada 2025"""
        driver_stats = {
            # Datos aproximados basados en rendimiento real 2025
            'VER': {'avg_quali': 8, 'avg_grid': 8, 'avg_position': 8, 'season_points': 180, 'gap_to_pole': 0.8, 'fp1_gap': 0.6},
            'PER': {'avg_quali': 12, 'avg_grid': 12, 'avg_position': 12, 'season_points': 45, 'gap_to_pole': 1.2, 'fp1_gap': 1.0},
            
            'NOR': {'avg_quali': 3, 'avg_grid': 3, 'avg_position': 4, 'season_points': 280, 'gap_to_pole': 0.2, 'fp1_gap': 0.3},
            'PIA': {'avg_quali': 5, 'avg_grid': 5, 'avg_position': 5, 'season_points': 220, 'gap_to_pole': 0.3, 'fp1_gap': 0.4},
            
            'LEC': {'avg_quali': 4, 'avg_grid': 4, 'avg_position': 5, 'season_points': 250, 'gap_to_pole': 0.3, 'fp1_gap': 0.4},
            'SAI': {'avg_quali': 7, 'avg_grid': 7, 'avg_position': 7, 'season_points': 160, 'gap_to_pole': 0.6, 'fp1_gap': 0.5},
            
            'HAM': {'avg_quali': 6, 'avg_grid': 6, 'avg_position': 6, 'season_points': 190, 'gap_to_pole': 0.5, 'fp1_gap': 0.4},
            'RUS': {'avg_quali': 7, 'avg_grid': 7, 'avg_position': 7, 'season_points': 170, 'gap_to_pole': 0.6, 'fp1_gap': 0.5},
            
            'ALO': {'avg_quali': 9, 'avg_grid': 9, 'avg_position': 9, 'season_points': 80, 'gap_to_pole': 0.9, 'fp1_gap': 0.7},
            'STR': {'avg_quali': 14, 'avg_grid': 14, 'avg_position': 14, 'season_points': 15, 'gap_to_pole': 1.4, 'fp1_gap': 1.2},
            
            'ALB': {'avg_quali': 11, 'avg_grid': 11, 'avg_position': 11, 'season_points': 55, 'gap_to_pole': 1.1, 'fp1_gap': 0.9},
            'GAS': {'avg_quali': 13, 'avg_grid': 13, 'avg_position': 13, 'season_points': 25, 'gap_to_pole': 1.3, 'fp1_gap': 1.1},
            'OCO': {'avg_quali': 15, 'avg_grid': 15, 'avg_position': 15, 'season_points': 10, 'gap_to_pole': 1.5, 'fp1_gap': 1.3},
            
            'HUL': {'avg_quali': 12, 'avg_grid': 12, 'avg_position': 12, 'season_points': 30, 'gap_to_pole': 1.2, 'fp1_gap': 1.0},
            'MAG': {'avg_quali': 16, 'avg_grid': 16, 'avg_position': 16, 'season_points': 8, 'gap_to_pole': 1.6, 'fp1_gap': 1.4},
            
            'TSU': {'avg_quali': 17, 'avg_grid': 17, 'avg_position': 17, 'season_points': 5, 'gap_to_pole': 1.7, 'fp1_gap': 1.5},
            'LAW': {'avg_quali': 18, 'avg_grid': 18, 'avg_position': 18, 'season_points': 2, 'gap_to_pole': 1.8, 'fp1_gap': 1.6},
            
            'ZHO': {'avg_quali': 19, 'avg_grid': 19, 'avg_position': 19, 'season_points': 1, 'gap_to_pole': 1.9, 'fp1_gap': 1.7},
            'BOT': {'avg_quali': 20, 'avg_grid': 20, 'avg_position': 20, 'season_points': 0, 'gap_to_pole': 2.0, 'fp1_gap': 1.8},
        }
        
        return driver_stats.get(driver, {
            'avg_quali': 10, 'avg_grid': 10, 'avg_position': 10, 
            'season_points': 0, 'gap_to_pole': 1.0, 'fp1_gap': 0.8
        })
    
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
        
        print(f"{'Pos':<4} {'Piloto':<6} {'Equipo':<16} {'Tipo':<30} {'Fuente':<12}")
        print("-" * 100)
        
        for _, row in predictions_df.iterrows():
            source = row.get('prediction_source', '📊 Config')
            
            print(f"P{row['final_position']:<3} {row['driver']:<6} {row['team']:<16} "
                  f"{row['driver_type']:<30} {source:<12}")
        
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