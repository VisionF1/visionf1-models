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
        # Cache para estadísticas de equipos 2025 calculadas desde datos reales
        self._team_2025_stats = None
        # Cache para estadísticas de pilotos 2025 calculadas desde datos reales
        self._driver_2025_stats = None
        # Mostrar línea por piloto tras predecir (configurable por env)
        self.print_each = os.getenv("VISIONF1_PRINT_EACH", "1").strip().lower() in ("1", "true", "yes", "on")
        # Debug detallado de cálculo de stats de equipos (default ON)
        self.team_stats_debug = os.getenv("VISIONF1_TEAM_DEBUG", "1").strip().lower() in ("1", "true", "yes", "on")
        self.enhanced_data_preparer = EnhancedDataPreparer(quiet=True)
        # Debug de features por piloto (guardar última fila de features preparada)
        self._last_features_by_driver = {}
        self._feature_debug_names = {"PIA", "NOR"}
        # Cargar metadata de entrenamiento (features) para auditoría en inferencia
        self._trained_feature_names = self._load_trained_feature_names()
        if self._trained_feature_names:
            try:
                print(f"\n🧾 Inferencia: FEATURES DEL MODELO ({len(self._trained_feature_names)})")
                # Imprimir compactas en una línea por brevedad
                print("   " + ", ".join(self._trained_feature_names))
                # Auditoría rápida pre-race vs post-race sospechoso
                self._audit_trained_features(self._trained_feature_names)
                # Mostrar resumen de auditoría si existe
                audit_file = "app/models_cache/feature_audit.txt"
                if os.path.exists(audit_file):
                    print(f"   📄 Auditoría previa encontrada: {audit_file}")
            except Exception:
                pass

    def _align_to_trained(self, X_df: pd.DataFrame) -> pd.DataFrame:
        try:
            trained_cols = getattr(self, '_trained_feature_names', None)
            if not trained_cols:
                return X_df
            X_df = X_df.copy()
            # columnas faltantes -> 0.0
            missing = [c for c in trained_cols if c not in X_df.columns]
            for c in missing:
                X_df[c] = 0.0
            # descartar extras y reordenar
            X_df = X_df[trained_cols]
            return X_df
        except Exception:
            return X_df
        
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
            # 1. Sin overfitting relevante (score < 1.2)
            # 2. Buen CV score (menor es mejor)
            # 3. R² en test razonable
            
            if overfitting_score < 1.2 and test_r2 >= 0.3:
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
    
    def predict_positions_2025(self, recent_data_2024=None, force_model_name: str | None = None):
        """Predice posiciones priorizando información de 2025.
        Si force_model_name está definido en {'RandomForest','XGBoost','GradientBoosting'},
        se carga ese modelo. Caso contrario, se selecciona automáticamente por métricas.
        """
        print("🎯 Prediciendo posiciones")

        # Intentar cargar modelo entrenado (o uno específico si se solicita)
        if force_model_name:
            print(f"🎯 Forzando uso de modelo: {force_model_name}")
            model_loaded = self._load_specific_model(force_model_name)
            if not model_loaded:
                print("⚠️  No se pudo cargar el modelo forzado; usando selección por métricas")
                model_loaded = self.load_best_model()
        else:
            model_loaded = self.load_best_model()

        # 0) Construir base_data para TODOS los pilotos en una sola tabla
        current_race_number = PREDICTION_CONFIG["next_race"].get("race_number", 1)
        base_rows = []
        for d, cfg in self.drivers_config.items():
            try:
                row = self._build_base_data_row(d, cfg)
                # Usar el código de piloto como índice para mantener mapping estable
                row.index = [d]
                base_rows.append(row)
            except Exception as e:
                print(f"⚠️  Error creando base row para {d}: {e}")
        base_df = pd.concat(base_rows, axis=0) if base_rows else pd.DataFrame()

        # 1) Preparar features avanzadas en lote (para activar rankings relativos por carrera)
        X, y, _, feature_names = (None, None, None, None)
        if not base_df.empty:
            try:
                prep = self.enhanced_data_preparer.prepare_enhanced_features(base_df.copy(), inference=True)
                if isinstance(prep, tuple) and len(prep) == 4:
                    X, y, _, feature_names = prep
                    if X is not None and not X.empty and base_df is not None and len(base_df) == len(X):
                        X.index = base_df.index  # Asegurar que los índices coincidan

                    X = self._align_to_trained(X)
            except Exception as e:
                print(f"⚠️  Error preparando features en lote: {e}")

        predictions = []
        # DEBUG: Guardar features por piloto para comparar luego
        debug_features = {}
        for driver, config in self.drivers_config.items():
            try:
                # 2) Construir features alineadas al esquema entrenado (priorizar lote; si no, 1x piloto)
                ml_features = self._get_aligned_features(driver, X, base_df)
                if ml_features is None:
                    # Intento 2: construir a mano según el esquema entrenado usando base_df
                    try:
                        ml_features = self._build_aligned_from_base(driver, base_df)
                    except Exception:
                        ml_features = None
                if ml_features is None:
                    # Fallback definitivo a features básicas si no se pudo obtener/alinéar
                    ml_features = self._create_basic_ml_features(driver, config)

                # Guardar features para debug
                debug_features[driver] = ml_features

                # 3) Predecir con el modelo ML o fallback
                use_ml = (model_loaded and self.trained_model is not None and ml_features is not None)
                # Validar dimensión contra las features entrenadas (si están disponibles)
                try:
                    if use_ml and hasattr(self, '_trained_feature_names') and self._trained_feature_names:
                        use_ml = (len(ml_features) == len(self._trained_feature_names))
                        if not use_ml:
                            print(f"⚠️  Dimensión de features inesperada para {driver}: {len(ml_features)} != {len(self._trained_feature_names)}; usando fallback")
                except Exception:
                    pass

                if use_ml:
                    ml_prediction = self.trained_model.predict([ml_features])[0]
                    predicted_position = self._convert_ml_to_position(ml_prediction)
                    prediction_source = "ML-100%"
                else:
                    predicted_position = 10.0 + random.uniform(-3.0, 3.0)
                    prediction_source = "Fallback"

                # 4) Penalización y etapa: rookies vs cambios de equipo
                if config.get("rookie", False):
                    predicted_position += self.penalties.get("rookie", 2.5)
                    driver_type = "🆕 Rookie"
                elif config.get("team_change", False):
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

                # 5) Limitar entre 1 y 20 y loguear
                predicted_position = max(1, min(20, predicted_position))
                if self.print_each:
                    team = config.get("team", "?")
                    print(f"   🔹 {driver} ({team}): P{predicted_position:.10f} · {driver_type} · {prediction_source}")

                predictions.append({
                    'driver': driver,
                    'team': config["team"],
                    'predicted_position': predicted_position,
                    'driver_type': driver_type,
                    'prediction_source': prediction_source
                })

            except Exception as e:
                print(f"⚠️  Error prediciendo {driver}: {e}")
                base_position = 10.0 + random.uniform(-5.0, 5.0)
                predictions.append({
                    'driver': driver,
                    'team': config["team"],
                    'predicted_position': max(1, min(20, base_position)),
                    'driver_type': "⚠️ Error",
                    'prediction_source': "� Fallback"
                })

        # DEBUG: Comparar features de Piastri y Norris
        if 'PIA' in debug_features and 'NOR' in debug_features:
            print("\n🔬 DEBUG: Comparación de features entre Piastri y Norris:")
            print("PIA:", debug_features['PIA'])
            print("NOR:", debug_features['NOR'])
            if np.array_equal(debug_features['PIA'], debug_features['NOR']):
                print("⚠️  Los vectores de features son idénticos para Piastri y Norris.")
            else:
                print("✅ Los vectores de features son distintos para Piastri y Norris.")

        # Crear DataFrame y aplicar adaptación progresiva
        df = pd.DataFrame(predictions)
        # Resolver empates exactos en predicted_position con desempate determinístico
        try:
            if 'predicted_position' in df.columns and not df.empty:
                # Agrupar por valor redondeado para detectar empates prácticos
                df['_pp_round'] = df['predicted_position'].round(4)
                for val, grp in df.groupby('_pp_round'):
                    if len(grp) > 1:
                        # Ordenar mejor a peor usando señales pre-race estables
                        # Preferir mayor driver_competitiveness y points_last_3
                        def score(row):
                            drv = row['driver']
                            s1 = float(base_df.loc[drv, 'driver_competitiveness']) if drv in base_df.index and 'driver_competitiveness' in base_df.columns else 0.0
                            s2 = float(base_df.loc[drv, 'points_before_race']) if drv in base_df.index and 'points_before_race' in base_df.columns else 0.0
                            return (s1, s2)
                        order = grp.apply(score, axis=1).apply(lambda x: (-x[0], -x[1]))
                        # argsort por la tupla (desc)
                        order_idx = order.sort_values().index
                        # Aplicar pequeños desplazamientos determinísticos para romper el empate
                        eps = 0.001
                        for rank, idx in enumerate(order_idx):
                            df.loc[idx, 'predicted_position'] = df.loc[idx, 'predicted_position'] + (rank * eps)
                df.drop(columns=['_pp_round'], inplace=True)
        except Exception:
            pass
        # Guardar predicción cruda antes de aplicar penalizaciones progresivas
        if 'predicted_position' in df.columns and 'predicted_position_raw' not in df.columns:
            df['predicted_position_raw'] = df['predicted_position']
        
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
    
    def _get_aligned_features(self, driver, X_batch, base_df):

        """Obtiene el vector de features para un piloto alineado al esquema entrenado.
        Preferimos usar la fila del lote X (consistencia de rankings). Si no existe,
        generamos 1x piloto con el preparador avanzado. Reordenamos/llenamos columnas
        según self._trained_feature_names cuando están disponibles.
        """
        try:
            # 1) Usar fila del lote si está disponible
            if X_batch is not None and driver in X_batch.index:
                feat_df = X_batch.loc[[driver]].copy()

                # Debug selectivo
                try:
                    self._last_features_by_driver[driver] = feat_df
                    if driver in self._feature_debug_names:
                        print(f"\n🧩 [debug] Features preparadas (lote) para {driver}:")
                        with pd.option_context('display.max_columns', None, 'display.width', 200):
                            print(feat_df.reset_index(drop=True).to_string(index=False))
                except Exception:
                    pass

                # 🔧 Alinear al orden entrenado
                feat_df = self._align_to_trained(feat_df)
                return feat_df.iloc[0].values

            # 2) Construir features solo para este piloto (inference=True reusa encoders)
            if base_df is not None and not base_df.empty and driver in base_df.index:
                one_base = base_df.loc[[driver]].copy()
                try:
                    prep = self.enhanced_data_preparer.prepare_enhanced_features(one_base, inference=True)
                    if isinstance(prep, tuple) and len(prep) == 4 and prep[0] is not None and not prep[0].empty:
                        feat_df = prep[0]
                        # si no quedó indexado por driver, tomamos 1ª fila
                        if driver in feat_df.index:
                            feat_df = feat_df.loc[[driver]].copy()
                        else:
                            feat_df = feat_df.iloc[[0]].copy()
                        # 🔧 Alinear al orden entrenado
                        feat_df = self._align_to_trained(feat_df)
                        return feat_df.iloc[0].values
                except Exception:
                    return None

        except Exception:
            return None
        return None

    def _build_aligned_from_base(self, driver: str, base_df: pd.DataFrame):
        """Construye un vector alineado al orden de self._trained_feature_names usando la fila base.
        Usa valores reales cuando existan, y defaults seguros cuando falten.
        """
        if not getattr(self, '_trained_feature_names', None):
            return None
        if base_df is None or base_df.empty or driver not in base_df.index:
            return None
        row = base_df.loc[driver]
        # Helper para obtener valores con default
        def g(col, default):
            try:
                v = row.get(col, default)
                if pd.isna(v):
                    return default
                return float(v)
            except Exception:
                return default
        # Encoder de pilotos (si existe)
        try:
            enc_val = None
            if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
                if driver in getattr(self.label_encoder, 'classes_', []):
                    enc_val = float(self.label_encoder.transform([driver])[0])
            driver_encoded = enc_val if enc_val is not None else 0.0
        except Exception:
            driver_encoded = 0.0

        # Construir mapa de valores base conocidos
        vals = {
            'driver_encoded': driver_encoded,
            'team_encoded': 0.0,
            'driver_skill_factor': g('driver_competitiveness', 0.75),
            'team_strength_factor': g('team_competitiveness', 0.65),
            'driver_team_synergy': g('driver_competitiveness', 0.75) * g('team_competitiveness', 0.65),
            'driver_competitiveness': g('driver_competitiveness', 0.75),
            'team_competitiveness': g('team_competitiveness', 0.65),
            'session_air_temp': g('session_air_temp', 25.0),
            'session_track_temp': g('session_track_temp', 35.0),
            'session_humidity': g('session_humidity', 60.0),
            'session_rainfall': g('session_rainfall', 0.0),
            'driver_weather_skill': g('driver_weather_skill', 0.75),
            'driver_rain_advantage': g('driver_weather_skill', 0.75) * g('session_rainfall', 0.0),
            'fp1_gap_to_fastest': g('fp1_gap_to_fastest', 1.0),
            'points_last_3': 5.0,
            'heat_index': 0.5,
            'weather_difficulty_index': 0.5,
            'team_track_avg_position': 10.0,
            'sector_consistency': 0.5,
            'overtaking_ability': g('overtaking_ability', 0.0),
            'total_laps': 58.0,
            'fp3_best_time': g('fp3_best_time', 90.0),
            'fp3_rank': g('fp3_rank', 10.0),
        }
        # Ordenar según el esquema entrenado y rellenar faltantes con 0.0
        ordered = []
        for name in self._trained_feature_names:
            ordered.append(float(vals.get(name, 0.0)))
        return np.array(ordered, dtype=float)

    def _load_trained_feature_names(self):
        """Carga la lista de features usadas en entrenamiento si existe."""
        try:
            path = "app/models_cache/feature_names.pkl"
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    names = pickle.load(f)
                # Normalizar a lista simple de strings
                if isinstance(names, (list, tuple)):
                    return [str(x) for x in names]
                if hasattr(names, 'tolist'):
                    return [str(x) for x in names.tolist()]
                return [str(names)]
        except Exception as e:
            print(f"⚠️  No se pudieron cargar feature_names entrenadas: {e}")
        return []

    def _audit_trained_features(self, feature_names):
        """Clasifica features del modelo en pre-race seguras vs sospechosas post-race y las imprime."""
        try:
            if not feature_names:
                return
            unsafe_tokens = [
                'race_position', 'final_position', 'grid_position', 'grid_to_race_change',
                'quali_vs_race_delta', 'points_efficiency', 'fastest_lap', 'status',
                'race_best_lap', 'lap_time_std', 'lap_time_consistency'
            ]
            safe_prefixes = [
                'fp1_', 'fp2_', 'fp3_', 'session_', 'weather_', 'team_avg_position_',
                'driver_', 'team_', 'expected_grid_position', 'points_last_3',
                'avg_position_last_3', 'avg_quali_last_3', 'overtaking_ability',
                'team_track_avg_position', 'driver_track_avg_position', 'sector_',
                'heat_index', 'temp_deviation_from_ideal', 'weather_difficulty_index',
                'total_laps'
            ]
            unsafe, safe = [], []
            for name in feature_names:
                n = (name or '').lower()
                if n == 'fp3_best_time':
                    safe.append(name)
                elif 'quali_position' in n:
                    unsafe.append(name)
                elif any(tok in n for tok in unsafe_tokens):
                    unsafe.append(name)
                elif any(n.startswith(p) for p in safe_prefixes):
                    safe.append(name)
                else:
                    safe.append(name)
            if unsafe:
                print("   🚫 Posibles post-race (revisar y eliminar para pre-race puro):")
                print("      - " + "\n      - ".join(sorted(unsafe)))
            print("   ✅ Pre-race (ok):")
            print("      - " + "\n      - ".join(sorted(safe)))
        except Exception:
            pass

    def _build_base_data_row(self, driver, config):
        """Crea una fila base de datos para un piloto (sin features derivadas)."""
        # Obtener carrera actual de la configuración
        current_race_name = PREDICTION_CONFIG["next_race"].get("race_name", "Hungarian Grand Prix")
        current_race_number = PREDICTION_CONFIG["next_race"].get("race_number", 13)

        # Generar datos realistas específicos por piloto
        driver_pace = self._estimate_driver_pace(driver, config)

        # Datos específicos por piloto basados en rendimiento y equipo
        team_performance = self._get_team_performance_estimate(config["team"])
        driver_performance = self._get_driver_performance_estimate(driver)

        # Derivados simples por piloto/equipo
        team_comp = max(0.3, min(0.95, team_performance.get('competitiveness', 6.0) / 10.0))
        drv_avg_pos = float(driver_performance.get('avg_position', 10))
        driver_comp = max(0.3, min(0.95, 1.0 - (drv_avg_pos / 22.0)))
        driver_weather_skill = round(max(0.3, min(0.95, driver_comp * 0.9)), 3)

        # Pequeño offset determinístico por piloto
        driver_sig = (sum(ord(c) for c in driver) % 97) / 1000.0
        fp3_best_time = max(30.0, driver_pace - 0.35 + driver_sig)
        fp3_rank = max(1, int(round(driver_performance.get('avg_quali', 10) + (driver_sig * 2 - 0.1))))
        overtaking_ability = float(max(-5.0, min(5.0, driver_performance.get('avg_quali', 10) - drv_avg_pos)))

        base_data = pd.DataFrame({
            'driver': [driver],
            # Usar el nombre correcto esperado por el FE avanzado
            'quali_position': [driver_performance.get('avg_quali', 10)],
            'grid_position': [driver_performance.get('avg_grid', 10)],
            'team': [config["team"]],
            'session_type': ['Race'],
            'points_before_race': [driver_performance.get('season_points', 0)],
            'race_name': [current_race_name],
            'season': [2025],
            'year': [2025],
            'round': [current_race_number],

            # Meteo
            'session_air_temp': [self._get_weather_value('session_air_temp')],
            'session_track_temp': [self._get_weather_value('session_track_temp')],
            'session_humidity': [self._get_weather_value('session_humidity')],
            'session_rainfall': [self._get_weather_value('session_rainfall')],

            # Piloto
            'quali_best_time': [driver_pace + 0.5],
            'race_best_lap_time': [driver_pace + 1.0],
            'clean_air_pace': [driver_pace],
            'quali_gap_to_pole': [driver_performance.get('gap_to_pole', 1.0)],
            'fp1_gap_to_fastest': [driver_performance.get('fp1_gap', 0.5)],
            'fp3_best_time': [fp3_best_time],
            'fp3_rank': [fp3_rank],
            # Eliminado expected_grid_position para evitar fuga post-quali
            'overtaking_ability': [overtaking_ability],
            'driver_competitiveness': [round(driver_comp, 3)],
            'team_competitiveness': [round(team_comp, 3)],
            'driver_weather_skill': [driver_weather_skill],

            # Circuito y tiempos de vuelta estimados
            'sector1_time': [driver_pace * 0.32],
            'sector2_time': [driver_pace * 0.38],
            'sector3_time': [driver_pace * 0.30],
            'lap_time': [driver_pace],
            'position': [driver_performance.get('avg_position', 10)],
            'fastest_lap': [False],
            'status': ['Finished']
        })

        # Logs de depuración selectivos
        if driver in {"VER", "HAD", "ANT"} or driver in self._feature_debug_names:
            print(f"\n🔍 [debug] Base data para {driver} ({config['team']}):")
            print(base_data.to_string(index=False))
            print("   ↳ team_performance:", team_performance)
            print("   ↳ driver_performance:", driver_performance)

        return base_data
    
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

            # Derivados simples y estables por piloto/equipo para evitar defaults idénticos
            team_comp = max(0.3, min(0.95, team_performance.get('competitiveness', 6.0) / 10.0))
            drv_avg_pos = float(driver_performance.get('avg_position', 10))
            driver_comp = max(0.3, min(0.95, 1.0 - (drv_avg_pos / 22.0)))
            # Habilidad en lluvia aproximada a partir de la competitividad del piloto
            driver_weather_skill = round(max(0.3, min(0.95, driver_comp * 0.9)), 3)
            # Estimación FP3
            # Pequeño offset determinístico por piloto (estable entre ejecuciones)
            driver_sig = (sum(ord(c) for c in driver) % 97) / 1000.0
            fp3_best_time = max(30.0, driver_pace - 0.35 + driver_sig)
            fp3_rank = max(1, int(round(driver_performance.get('avg_quali', 10) + (driver_sig * 2 - 0.1))))
            # Capacidad de sobrepaso (gana posiciones si corre mejor que clasifica)
            overtaking_ability = float(max(-5.0, min(5.0, driver_performance.get('avg_quali', 10) - drv_avg_pos)))
            
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
                'fp3_best_time': [fp3_best_time],
                'fp3_rank': [fp3_rank],
                # Eliminado expected_grid_position para evitar fuga post-quali
                'overtaking_ability': [overtaking_ability],
                'driver_competitiveness': [round(driver_comp, 3)],
                'team_competitiveness': [round(team_comp, 3)],
                'driver_weather_skill': [driver_weather_skill],
                
                # Datos específicos por circuito y piloto
                'sector1_time': [driver_pace * 0.32],  # ~32% del tiempo total
                'sector2_time': [driver_pace * 0.38],  # ~38% del tiempo total
                'sector3_time': [driver_pace * 0.30],  # ~30% del tiempo total
                'lap_time': [driver_pace],
                'position': [driver_performance.get('avg_position', 10)],
                'fastest_lap': [False],
                'status': ['Finished']
            })
            if driver in {"VER", "HAD", "ANT"} or driver in self._feature_debug_names:
                print(f"\n🔍 [debug] Base data para {driver} ({config['team']}):")
                print(base_data.to_string(index=False))
                print("   ↳ team_performance:", team_performance)
                print("   ↳ driver_performance:", driver_performance)
            # Usar el sistema avanzado de features
            result = self.enhanced_data_preparer.prepare_enhanced_features(base_data)
            
            # El método prepare_enhanced_features devuelve 4 elementos: X, y, label_encoder, feature_names
            if isinstance(result, tuple) and len(result) == 4:
                X, _, _, feature_names = result  # También recibimos nombres
            elif isinstance(result, tuple) and len(result) == 2:
                X, _ = result  # Formato alternativo (X, y)
            else:
                X = result     # Formato solo X
            
            # Debug: mostrar la fila de features preparada para PIA/NOR (o seleccionados)
            if (driver in self._feature_debug_names) and X is not None and len(X) > 0:
                try:
                    if hasattr(X, 'columns'):
                        names = list(X.columns)
                        vals = X.iloc[0].tolist()
                    else:
                        # Si X no tiene columnas, intentamos usar feature_names
                        names = feature_names if 'feature_names' in locals() and feature_names else [f'f{i}' for i in range(len(X[0]))]
                        vals = list(X[0]) if hasattr(X, '__getitem__') else []
                    feat_df = pd.DataFrame([vals], columns=names)
                    self._last_features_by_driver[driver] = feat_df
                    print(f"\n🧩 [debug] Features preparadas para {driver} (primer fila):")
                    # Mostrar todas las columnas y valores de esa fila
                    with pd.option_context('display.max_columns', None, 'display.width', 200):
                        print(feat_df.to_string(index=False))
                except Exception as _e:
                    print(f"⚠️ [debug] No se pudieron mostrar features para {driver}: {_e}")

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
        """Estima rendimiento del equipo basándose en resultados reales 2025.

        Devuelve un dict con:
          - avg_position: media de posiciones de carrera 2025 (menor es mejor)
          - competitiveness: [0..10], inversa de avg_position normalizada entre equipos
        """
        # Construir cache si no existe
        if self._team_2025_stats is None:
            if self.team_stats_debug:
                print("🔎 [team-stats] Construyendo estadísticas 2025 desde cache...")
            self._team_2025_stats = self._compute_team_stats_2025()
            if self.team_stats_debug:
                size = 0 if not self._team_2025_stats else len(self._team_2025_stats)
                print(f"✅ [team-stats] Construidas: {size} equipos")

        stats = self._team_2025_stats or {}

        # Intentar obtener directo
        if team in stats:
            if self.team_stats_debug:
                print(f"   ↪️ [team-stats] Usando stats directas para '{team}': {stats[team]}")
            return stats[team]

        # Resolver alias comunes
        alias = {
            'Haas': 'Haas F1 Team',
            'Sauber': 'Kick Sauber',
            'Alfa Romeo': 'Kick Sauber',
            'RB': 'Racing Bulls',
            'AlphaTauri': 'Racing Bulls',
            'Alpha Tauri': 'Racing Bulls',
            'Red Bull': 'Red Bull Racing',
            'Williams Racing': 'Williams',
            'McLaren F1 Team': 'McLaren',
            'BWT Alpine F1 Team': 'Alpine',
            'Aston Martin Aramco Cognizant F1 Team': 'Aston Martin',
            'Mercedes-AMG Petronas F1 Team': 'Mercedes',
        }
        mapped = alias.get(team)
        if mapped and mapped in stats:
            if self.team_stats_debug:
                print(f"   🔁 [team-stats] Alias '{team}' → '{mapped}': {stats[mapped]}")
            return stats[mapped]

        # Fallback conservador
        if self.team_stats_debug:
            print(f"   ⚠️ [team-stats] Fallback para '{team}': avg_position=10.0 competitividad=5.0")
        return {'avg_position': 10.0, 'competitiveness': 5.0}

    def _compute_team_stats_2025(self):
        """Calcula tabla de equipos 2025 desde cache raw_data y deriva competitividad.

        competitive = inverse-minmax(avg_position) escalado a [0..10].
        """
        try:
            import os
            import pickle
            import pandas as pd
            from app.core.utils.team_mapping_utils import quick_team_mapping

            cache_dir = "app/models_cache/raw_data"
            rows = []
            files_found = 0
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    if "2025" in filename and filename.endswith('.pkl'):
                        files_found += 1
                        if self.team_stats_debug:
                            print(f"   📦 [team-stats] Procesando archivo: {filename}")
                        try:
                            filepath = os.path.join(cache_dir, filename)
                            with open(filepath, 'rb') as f:
                                race_data = pickle.load(f)
                            if self.team_stats_debug:
                                print(f"      • tipo={type(race_data)}")
                            temp = []
                            # Dict de pilotos -> atributos o dict con DataFrame en 'data'
                            if isinstance(race_data, dict):
                                df_in = race_data.get('data') if 'data' in race_data else None
                                if df_in is not None and hasattr(df_in, 'columns'):
                                    # Caso: {'data': DataFrame, 'metadata': ...}
                                    if self.team_stats_debug:
                                        try:
                                            print(f"      • detectado DataFrame en 'data' con columnas: {list(df_in.columns)}")
                                        except Exception:
                                            pass
                                    df0 = df_in.copy()
                                    team_col = next((c for c in ['team', 'Team', 'constructor', 'Constructor', 'constructor_name', 'ConstructorName'] if c in df0.columns), None)
                                    pos_col = next((c for c in ['race_position', 'position', 'Position', 'final_position', 'FinishPosition', 'ResultPosition', 'race_result_position'] if c in df0.columns), None)
                                    drv_col = next((c for c in ['driver', 'Driver', 'code', 'Code', 'DriverNumber', 'DriverCode'] if c in df0.columns), None)
                                    if team_col and pos_col:
                                        cols = [col for col in [drv_col, team_col, pos_col] if col is not None]
                                        s = df0[cols].copy()
                                        # Renombrar a estándar
                                        new_cols = []
                                        for idx, col in enumerate(['driver', 'team', 'race_position'][:s.shape[1]]):
                                            new_cols.append(col)
                                        s.columns = new_cols
                                        s['race_position'] = pd.to_numeric(s['race_position'], errors='coerce')
                                        s = s.dropna(subset=['team', 'race_position'])
                                        for _, r in s.iterrows():
                                            temp.append({'driver': r.get('driver', 'UNK'), 'team': r['team'], 'race_position': float(r['race_position'])})
                                else:
                                    # Dict genérico: iterar clave por clave
                                    for drv, d in race_data.items():
                                        if isinstance(d, dict):
                                            team_val = d.get('team') or d.get('Team') or d.get('constructor') or d.get('Constructor')
                                            pos_val = d.get('race_position') or d.get('position') or d.get('Position') or d.get('final_position')
                                            if team_val is not None and isinstance(pos_val, (int, float)):
                                                temp.append({'driver': drv, 'team': team_val, 'race_position': float(pos_val)})
                            # Lista de dicts
                            elif isinstance(race_data, list):
                                for rec in race_data:
                                    if isinstance(rec, dict):
                                        team_val = rec.get('team') or rec.get('Team') or rec.get('constructor') or rec.get('Constructor')
                                        pos_val = rec.get('race_position') or rec.get('position') or rec.get('Position') or rec.get('final_position')
                                        drv = rec.get('driver') or rec.get('Driver') or rec.get('code') or rec.get('Code') or 'UNK'
                                        if team_val is not None and isinstance(pos_val, (int, float)):
                                            temp.append({'driver': drv, 'team': team_val, 'race_position': float(pos_val)})
                            # DataFrame
                            elif isinstance(race_data, pd.DataFrame):
                                df0 = race_data.copy()
                                # Normalizar columnas posibles
                                team_col = next((c for c in ['team', 'Team', 'constructor', 'Constructor', 'constructor_name', 'ConstructorName'] if c in df0.columns), None)
                                pos_col = next((c for c in ['race_position', 'position', 'Position', 'final_position', 'FinishPosition', 'ResultPosition'] if c in df0.columns), None)
                                drv_col = next((c for c in ['driver', 'Driver', 'code', 'Code', 'DriverNumber', 'DriverCode'] if c in df0.columns), None)
                                if team_col and pos_col:
                                    s = df0[[col for col in [drv_col, team_col, pos_col] if col is not None]].copy()
                                    new_cols = []
                                    for idx, col in enumerate(['driver', 'team', 'race_position'][:s.shape[1]]):
                                        new_cols.append(col)
                                    s.columns = new_cols
                                    s['race_position'] = pd.to_numeric(s['race_position'], errors='coerce')
                                    s = s.dropna(subset=['team', 'race_position'])
                                    for _, r in s.iterrows():
                                        temp.append({'driver': r.get('driver', 'UNK'), 'team': r['team'], 'race_position': float(r['race_position'])})
                            if temp:
                                df = pd.DataFrame(temp)
                                df = quick_team_mapping(df)
                                if self.team_stats_debug:
                                    try:
                                        print(f"      ↳ [team-stats] Filas extraídas: {len(df)}, Equipos únicos: {df['team'].nunique()}")
                                    except Exception:
                                        pass
                                rows.append(df)
                            else:
                                if self.team_stats_debug:
                                    print("      ⚠️ [team-stats] No se extrajeron filas válidas de este archivo")
                        except Exception as e:
                            if self.team_stats_debug:
                                print(f"      ❌ [team-stats] Error leyendo {filename}: {e}")
                            continue
            if self.team_stats_debug:
                print(f"   📚 [team-stats] Archivos 2025 encontrados: {files_found}")
            if not rows:
                if self.team_stats_debug:
                    print("   ❌ [team-stats] No se hallaron datos 2025 en cache")
                return {}
            all_df = pd.concat(rows, ignore_index=True)
            if self.team_stats_debug:
                try:
                    print(f"   📈 [team-stats] Total filas: {len(all_df)}, Equipos: {all_df['team'].nunique()}")
                except Exception:
                    pass
            team_avg = all_df.groupby('team')['race_position'].mean().to_frame('avg_position').reset_index()
            s_min, s_max = team_avg['avg_position'].min(), team_avg['avg_position'].max()
            if s_max == s_min:
                team_avg['competitiveness'] = 5.0
            else:
                inv_norm = (s_max - team_avg['avg_position']) / (s_max - s_min)
                team_avg['competitiveness'] = (inv_norm * 10.0).clip(0.0, 10.0)
            team_avg['avg_position'] = team_avg['avg_position'].round(3)
            team_avg['competitiveness'] = team_avg['competitiveness'].round(2)
            if self.team_stats_debug:
                try:
                    print("   🧮 [team-stats] Tabla agregada (primeros 10):")
                    print(team_avg.head(10).to_string(index=False))
                    print(f"   🔢 [team-stats] avg_pos range: {s_min:.3f}..{s_max:.3f}")
                except Exception:
                    pass
            out = {row['team']: {'avg_position': float(row['avg_position']), 'competitiveness': float(row['competitiveness'])}
                   for _, row in team_avg.iterrows()}
            if self.team_stats_debug:
                print("   📦 [team-stats] Diccionario final de stats 2025:")
                try:
                    sorted_items = sorted(out.items(), key=lambda kv: kv[1]['avg_position'])
                    for name, vals in sorted_items:
                        print(f"      • {name:<16} -> avg_position={vals['avg_position']:.3f}, competitiveness={vals['competitiveness']:.2f}")
                except Exception:
                    print(out)
            return out
        except Exception as e:
            if self.team_stats_debug:
                print(f"   ❌ [team-stats] Error computando stats 2025: {e}")
            return {}
    
    def _get_driver_performance_estimate(self, driver):
        """Obtiene estadísticas del piloto 2025 desde cache; fallback a valores aproximados.

        Devuelve dict con claves:
          - avg_quali, avg_grid, avg_position
          - season_points (suma 2025 a la fecha)
          - gap_to_pole (media de gaps por carrera en quali)
          - fp1_gap (media de gaps por carrera en FP1)
        """
        # Construir cache si no existe
        if self._driver_2025_stats is None:
            if self.team_stats_debug:
                print("🔎 [driver-stats] Construyendo estadísticas 2025 por piloto desde cache...")
            self._driver_2025_stats = self._compute_driver_stats_2025()
            if self.team_stats_debug:
                size = 0 if not self._driver_2025_stats else len(self._driver_2025_stats)
                print(f"✅ [driver-stats] Construidas: {size} pilotos")

        stats = self._driver_2025_stats or {}
        if driver in stats:
            if self.team_stats_debug:
                print(f"   ↪️ [driver-stats] Usando stats para '{driver}': {stats[driver]}")
            return stats[driver]

        # Fallback aproximado si el piloto no fue encontrado
        approx = {
            'avg_quali': 10, 'avg_grid': 10, 'avg_position': 10,
            'season_points': 0, 'gap_to_pole': 1.0, 'fp1_gap': 0.8
        }
        return approx

    def _compute_driver_stats_2025(self):
        """Calcula estadísticas agregadas por piloto usando los pickles 2025 en raw_data.

        - avg_quali: media de quali_position
        - avg_grid: media de grid_position
        - avg_position: media de race_position
        - season_points: suma de points
        - gap_to_pole: media por carrera de (quali_best_time - min_quali_best_time)
        - fp1_gap: media por carrera de (fp1_best_time - min_fp1_best_time) o usando fp1_avg_time si falta
        """
        try:
            import os, pickle
            import pandas as pd

            cache_dir = "app/models_cache/raw_data"
            frames = []
            files_found = 0
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    if "2025" in filename and filename.endswith('.pkl'):
                        files_found += 1
                        if self.team_stats_debug:
                            print(f"   📦 [driver-stats] Procesando archivo: {filename}")
                        try:
                            with open(os.path.join(cache_dir, filename), 'rb') as f:
                                data = pickle.load(f)
                            df_in = None
                            if isinstance(data, dict) and 'data' in data and hasattr(data['data'], 'columns'):
                                df_in = data['data']
                                if self.team_stats_debug:
                                    try:
                                        print(f"      • detectado DataFrame en 'data' con columnas: {list(df_in.columns)}")
                                    except Exception:
                                        pass
                            elif isinstance(data, pd.DataFrame):
                                df_in = data
                            elif isinstance(data, list):
                                # intentar conformar DataFrame
                                df_in = pd.DataFrame(data)

                            if df_in is None or not hasattr(df_in, 'columns'):
                                if self.team_stats_debug:
                                    print("      ⚠️ [driver-stats] Estructura no compatible; se omite")
                                continue

                            # Seleccionamos columnas tolerantes a nombres
                            cols = df_in.columns
                            sel = {}
                            sel['driver'] = next((c for c in ['driver','Driver','code','Code','DriverCode'] if c in cols), None)
                            sel['race_name'] = next((c for c in ['race_name','RaceName'] if c in cols), None)
                            sel['year'] = next((c for c in ['year','season','Year','Season'] if c in cols), None)
                            sel['quali_pos'] = next((c for c in ['quali_position','qualifying_position','QualiPosition'] if c in cols), None)
                            sel['grid_pos'] = next((c for c in [
                                'grid_position','GridPosition',
                                'grid','Grid','grid_pos','GridPos',
                                'starting_grid_position','StartingGridPosition',
                                'start_position','StartPosition'
                            ] if c in cols), None)
                            sel['race_pos'] = next((c for c in ['race_position','position','RacePosition','FinalPosition'] if c in cols), None)
                            sel['points'] = next((c for c in ['points','Points'] if c in cols), None)
                            sel['quali_best'] = next((c for c in ['quali_best_time','QualiBestTime'] if c in cols), None)
                            sel['fp1_best'] = next((c for c in ['fp1_best_time','FP1BestTime'] if c in cols), None)
                            sel['fp1_avg'] = next((c for c in ['fp1_avg_time','FP1AvgTime'] if c in cols), None)

                            if not sel['driver'] or not sel['race_pos']:
                                if self.team_stats_debug:
                                    print("      ⚠️ [driver-stats] Faltan columnas clave; se omite")
                                continue

                            use_cols = [v for v in sel.values() if v]
                            df = df_in[use_cols].copy()
                            # Renombrar
                            rename_map = {sel['driver']:'driver'}
                            if sel['race_name']: rename_map[sel['race_name']] = 'race_name'
                            if sel['year']: rename_map[sel['year']] = 'year'
                            if sel['quali_pos']: rename_map[sel['quali_pos']] = 'quali_position'
                            if sel['grid_pos']: rename_map[sel['grid_pos']] = 'grid_position'
                            rename_map[sel['race_pos']] = 'race_position'
                            if sel['points']: rename_map[sel['points']] = 'points'
                            if sel['quali_best']: rename_map[sel['quali_best']] = 'quali_best_time'
                            if sel['fp1_best']: rename_map[sel['fp1_best']] = 'fp1_best_time'
                            if sel['fp1_avg']: rename_map[sel['fp1_avg']] = 'fp1_avg_time'
                            df = df.rename(columns=rename_map)

                            # Tipos numéricos
                            for c in ['quali_position','grid_position','race_position','points','quali_best_time','fp1_best_time','fp1_avg_time']:
                                if c in df.columns:
                                    df[c] = pd.to_numeric(df[c], errors='coerce')

                            # Identificar carrera
                            if 'race_name' not in df.columns:
                                df['race_name'] = os.path.splitext(filename)[0]
                            if 'year' not in df.columns:
                                df['year'] = 2025

                            frames.append(df)
                            if self.team_stats_debug:
                                try:
                                    print(f"      ↳ [driver-stats] Filas extraídas: {len(df)}")
                                except Exception:
                                    pass
                        except Exception as e:
                            if self.team_stats_debug:
                                print(f"      ❌ [driver-stats] Error leyendo {filename}: {e}")
                            continue
            if self.team_stats_debug:
                print(f"   📚 [driver-stats] Archivos 2025 encontrados: {files_found}")
            if not frames:
                if self.team_stats_debug:
                    print("   ❌ [driver-stats] No se hallaron datos 2025 en cache")
                return {}

            all_df = pd.concat(frames, ignore_index=True)

            # Calcular gaps por carrera
            def compute_gaps(group: pd.DataFrame):
                g = group.copy()
                # Corregir grid_position inválido por carrera (todos 20.0)
                if 'grid_position' in g.columns:
                    non_na = g['grid_position'].dropna()
                    if len(non_na) > 0:
                        uniq = pd.unique(non_na)
                        if len(uniq) == 1:
                            try:
                                only_val = float(uniq[0])
                            except Exception:
                                only_val = None
                            if only_val == 20.0:
                                if 'quali_position' in g.columns and g['quali_position'].notna().any():
                                    if self.team_stats_debug:
                                        try:
                                            rn = g['race_name'].iloc[0] if 'race_name' in g.columns else 'unknown'
                                            print(f"      🔧 [driver-stats] grid_position=20 para todos en '{rn}', usando quali_position como proxy")
                                        except Exception:
                                            pass
                                    g['grid_position'] = g['quali_position']
                                else:
                                    g['grid_position'] = np.nan
                    else:
                        # Todo NaN → usar quali_position si existe
                        if 'quali_position' in g.columns:
                            g['grid_position'] = g['quali_position']
                else:
                    # No existe columna grid_position → crear desde quali_position si posible
                    if 'quali_position' in g.columns:
                        g['grid_position'] = g['quali_position']

                # Gap a la pole: usar quali_best_time
                if 'quali_best_time' in g.columns and g['quali_best_time'].notna().any():
                    min_q = g['quali_best_time'].min()
                    g['gap_to_pole'] = g['quali_best_time'] - min_q
                else:
                    g['gap_to_pole'] = np.nan
                # FP1 gap: preferir fp1_best_time, sino fp1_avg_time
                fp1_series = None
                if 'fp1_best_time' in g.columns and g['fp1_best_time'].notna().any():
                    fp1_series = g['fp1_best_time']
                elif 'fp1_avg_time' in g.columns and g['fp1_avg_time'].notna().any():
                    fp1_series = g['fp1_avg_time']
                if fp1_series is not None:
                    min_fp1 = fp1_series.min()
                    g['fp1_gap'] = fp1_series - min_fp1
                else:
                    g['fp1_gap'] = np.nan
                return g

            all_df = all_df.groupby(['race_name','year'], as_index=False, group_keys=False).apply(compute_gaps)

            # Agregación por piloto
            agg = all_df.groupby('driver').agg(
                avg_quali=('quali_position','mean'),
                avg_grid=('grid_position','mean'),
                avg_position=('race_position','mean'),
                season_points=('points','sum'),
                gap_to_pole=('gap_to_pole','mean'),
                fp1_gap=('fp1_gap','mean')
            ).reset_index()

            # Limpiar y redondear
            for c in ['avg_quali','avg_grid','avg_position']:
                if c in agg.columns:
                    agg[c] = agg[c].round(2)
            for c in ['gap_to_pole','fp1_gap']:
                if c in agg.columns:
                    agg[c] = agg[c].round(3)
            if 'season_points' in agg.columns:
                agg['season_points'] = agg['season_points'].fillna(0).round(1)

            if self.team_stats_debug:
                try:
                    print("   🧮 [driver-stats] Tabla agregada (primeros 10):")
                    print(agg.head(10).to_string(index=False))
                except Exception:
                    pass

            out = {
                row['driver']: {
                    'avg_quali': float(row['avg_quali']) if not pd.isna(row['avg_quali']) else 10.0,
                    'avg_grid': float(row['avg_grid']) if not pd.isna(row['avg_grid']) else 10.0,
                    'avg_position': float(row['avg_position']) if not pd.isna(row['avg_position']) else 10.0,
                    'season_points': float(row['season_points']) if not pd.isna(row['season_points']) else 0.0,
                    'gap_to_pole': float(row['gap_to_pole']) if not pd.isna(row['gap_to_pole']) else 1.0,
                    'fp1_gap': float(row['fp1_gap']) if not pd.isna(row['fp1_gap']) else 0.8,
                }
                for _, row in agg.iterrows()
            }

            if self.team_stats_debug:
                print("   📦 [driver-stats] Diccionario final de stats 2025 (primeros 10):")
                try:
                    shown = 0
                    for name, vals in out.items():
                        print(f"      • {name:<4} -> avg_q={vals['avg_quali']}, grid={vals['avg_grid']}, race={vals['avg_position']}, pts={vals['season_points']}, q_gap={vals['gap_to_pole']}, fp1_gap={vals['fp1_gap']}")
                        shown += 1
                        if shown >= 10:
                            break
                except Exception:
                    print(out)
            return out
        except Exception as e:
            if self.team_stats_debug:
                print(f"   ❌ [driver-stats] Error computando stats 2025: {e}")
            return {}
    
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
        
        print(f"{'Pos':<4} {'PredRaw':<8} {'Pred':<6} {'Piloto':<6} {'Equipo':<16} {'Tipo':<30} {'Fuente':<12}")
        print("-" * 100)
        
        for _, row in predictions_df.iterrows():
            source = row.get('prediction_source', '📊 Config')
            # Mostrar posiciones: cruda (antes de adaptar) y final (post-adaptación numérica)
            pred_raw_val = row.get('predicted_position_raw', np.nan)
            pred_val = row.get('predicted_position', np.nan)
            pred_raw_str = f"P{pred_raw_val:.2f}" if pd.notna(pred_raw_val) else "P--"
            pred_str = f"P{pred_val:.2f}" if pd.notna(pred_val) else "P--"
            
            print(f"P{row['final_position']:<3} {pred_raw_str:<8} {pred_str:<6} {row['driver']:<6} {row['team']:<16} "
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