import pickle
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit, KFold
import numpy as np
from app.core.predictors.random_forest import RandomForestPredictor
from app.core.predictors.xgboost_model import XGBoostPredictor
from app.core.predictors.gradient_boosting import GradientBoostingPredictor
from app.config import DATA_IMPORTANCE
from pathlib import Path
import json

INFERENCE_MANIFEST_PATH = Path("app/models_cache/inference_manifest.json")
CATEGORICAL_COLS = ["driver", "team", "race_name", "circuit_type"]


class ModelTrainer:
    def __init__(self, use_time_series_cv=False):
        self.use_time_series_cv = use_time_series_cv
        self.models = self._initialize_models_with_hyperparams()
        self.results = {}

    def _initialize_models_with_hyperparams(self):
        return {
            'RandomForest': {
                'model_class': RandomForestPredictor,
                'param_grid': {
                    'n_estimators': [50, 100],
                    'max_depth': [2, 3],
                    'min_samples_split': [20, 30],
                    'min_samples_leaf': [10, 15],
                    'max_features': ['sqrt', 0.5],
                    'bootstrap': [True],            # Fijo
                    'oob_score': [True]             # Fijo - TOTAL: 2^5 = 32 combinaciones
                }
            },
            'XGBoost': {
                'model_class': XGBoostPredictor,
                'param_grid': {
                    'n_estimators': [200, 400],
                    'max_depth': [2, 3],
                    'learning_rate': [0.03, 0.06],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.6, 0.8],
                    'reg_alpha': [2.0, 4.0],
                    'reg_lambda': [2.0, 4.0],
                    'min_child_weight': [10, 20]
                }
            },
            'GradientBoosting': {
                'model_class': GradientBoostingPredictor,
                'param_grid': {
                    'n_estimators': [30, 50],
                    'max_depth': [2, 3],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.6, 0.8],
                    'min_samples_split': [20, 30],
                    'min_samples_leaf': [10, 14],
                    'max_features': ['sqrt', 0.5],
                    'validation_fraction': [0.1]    # Fijo - TOTAL: 2^7 = 128 combinaciones
                }
            }
        }

    def _save_inference_manifest(self, feature_names, results, selection_metric="kendall"):
        """
        Guarda un manifiesto para inferencia con:
        - feature_names (orden exacto)
        - encoders por columna categórica
        - mejor modelo según métrica de selección
        """
        # Seleccionar mejor modelo
        best = None
        if selection_metric == "kendall":
            # Mayor kendall_tau_test, con penalización por overfitting
            candidates = []
            for name, m in results.items():
                if "error" in m: 
                    continue
                score = (m.get("kendall_tau_test", -1.0) or -1.0) - 0.05 * max(0.0, m.get("overfitting_score", 1.0) - 1.0)
                candidates.append((score, name))
            if candidates:
                best = sorted(candidates, reverse=True)[0][1]
        else:
            # Default: menor cv_mse_mean y sin overfitting alto
            best = None
            best_cv = float('inf')
            for name, m in results.items():
                if "error" in m: 
                    continue
                if m.get("overfitting_score", 2.0) >= 1.3:
                    continue
                if m.get("cv_mse_mean", float('inf')) < best_cv:
                    best = name
                    best_cv = m["cv_mse_mean"]

        # Armar rutas de encoders por columna
        encoders = {}
        for col in CATEGORICAL_COLS:
            path = Path(f"app/models_cache/{col}_encoder.pkl")
            if path.exists():
                encoders[col] = str(path)
            else:
                encoders[col] = None

        manifest = {
            "feature_names": list(feature_names or []),
            "categorical_cols": CATEGORICAL_COLS,
            "encoders": encoders,
            "best_model_name": best,
        }
        INFERENCE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INFERENCE_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"💾 Manifiesto de inferencia guardado: {INFERENCE_MANIFEST_PATH}")



    def train_all_models(self, X_train, X_test, y_train, y_test, label_encoder, feature_names, df_original=None, train_indices=None):
        print("Entrenando modelos con Cross-Validation...")

        if X_train is None or y_train is None:
            print("❌ No hay datos para entrenar")
            return {}

        # 📅 CALCULAR PESOS POR AÑOS
        sample_weights = None
        if df_original is not None and 'year' in df_original.columns and train_indices is not None:
            print("📅 Aplicando pesos por años a los datos de entrenamiento...")
            # Usar los índices correctos para obtener los datos de entrenamiento
            df_train = df_original.iloc[train_indices]
            sample_weights = self.calculate_year_weights(df_train)
        elif df_original is not None and 'year' in df_original.columns:
            print("📅 Aplicando pesos por años (sin índices, asumiendo orden)...")
            # Fallback: asumir que los primeros registros son de entrenamiento
            df_train = df_original.iloc[:len(X_train)]
            sample_weights = self.calculate_year_weights(df_train)
        else:
            print("⚠️ No se encontraron datos de años, usando pesos uniformes")

        # 🛡️ DETECCIÓN Y CORRECCIÓN DE OVERFITTING
        print("🛡️ Aplicando estrategias anti-overfitting...")
        
        if len(X_train) < 50:
            print("🛑 Dataset muy pequeño: aplicando regularización extrema")
            self._apply_extreme_regularization()
        elif len(X_train) < 200:
            print("⚠️ Dataset pequeño: aplicando regularización fuerte")
            self._apply_strong_regularization()
        else:
            print("✅ Dataset adecuado: usando regularización estándar")

        self._save_metadata(label_encoder, feature_names)
        # 🔎 Auditoría rápida de features usadas para entrenar (pre-race vs post-race)
        try:
            self._audit_features(feature_names, target_name=getattr(y_train, 'name', None))
        except Exception as _e:
            print(f"⚠️  Auditoría de features falló: {_e}")
        cv_splitter = self._get_cv_splitter(X_train)

        for name, model_config in self.models.items():
            print(f"\n{'='*50}")
            print(f"🔍 ENTRENANDO {name} CON CROSS-VALIDATION")
            print(f"{'='*50}")

            self._train_single_model_with_cv(
                name, model_config, X_train, X_test, y_train, y_test, cv_splitter, sample_weights
            )

        self._show_detailed_comparison()
        self._save_training_results(self.results, label_encoder, feature_names)
        
        return self.results

    def _apply_extreme_regularization(self):
        """Aplicar regularización extrema para datasets muy pequeños"""
        print("   🔒 Aplicando regularización extrema...")
        
        # RandomForest ultra-conservador
        self.models['RandomForest']['param_grid'].update({
            'n_estimators': [10, 20],
            'max_depth': [2, 3],
            'min_samples_split': [20, 30],
            'min_samples_leaf': [10, 15],
            'max_features': ['sqrt']
        })
        
        # XGBoost ultra-conservador  
        self.models['XGBoost']['param_grid'].update({
            'n_estimators': [10, 20],
            'max_depth': [2, 3],
            'learning_rate': [0.01, 0.05],
            'reg_alpha': [1.0, 2.0],
            'reg_lambda': [1.0, 2.0],
            'min_child_weight': [5, 10]
        })
        
        # GradientBoosting ultra-conservador
        self.models['GradientBoosting']['param_grid'].update({
            'n_estimators': [10, 20],
            'max_depth': [2],
            'learning_rate': [0.01, 0.05],
            'min_samples_split': [25, 35],
            'min_samples_leaf': [12, 20]
        })

    def _apply_strong_regularization(self):
        """Aplicar regularización fuerte para datasets pequeños"""
        print("   🔐 Aplicando regularización fuerte...")
        
        # Reducir complejidad manteniendo exploración
        for model_name in self.models:
            param_grid = self.models[model_name]['param_grid']
            
            if 'max_depth' in param_grid:
                param_grid['max_depth'] = [2]  # Limitar más la profundidad
            if 'n_estimators' in param_grid:
                param_grid['n_estimators'] = [30, 50]  # Menos estimadores
            if 'min_samples_split' in param_grid:
                param_grid['min_samples_split'] = [25, 35]
            if 'min_samples_leaf' in param_grid:
                param_grid['min_samples_leaf'] = [12, 18]
            if model_name == 'GradientBoosting' and 'learning_rate' in param_grid:
                param_grid['learning_rate'] = [0.03, 0.05]

    def _get_cv_splitter(self, X_train):
        n_samples = len(X_train)
        # Usar KFold por defecto (sin dependencia temporal)
        n_splits = max(3, min(5, n_samples // 5))
        print(f"📊 Usando KFold con {n_splits} splits")
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _train_single_model_with_cv(self, name, model_config, X_train, X_test, y_train, y_test, cv_splitter, sample_weights=None):
        try:
            model_class = model_config['model_class']
            base_model = model_class()

            print(f"🔧 Optimizando hiperparámetros para {name}...")

            # Solo usar fit_params para modelos que los necesiten (sin XGBoost)
            fit_params = {}
            if hasattr(base_model.model, "early_stopping_rounds") and name != 'XGBoost':
                fit_params = {
                    "eval_set": [(X_test, y_test)],
                    "early_stopping_rounds": 10,
                    "verbose": False
                }

            if len(X_train) > 20:
                # CONFIGURACIÓN ESPECIAL PARA XGBOOST SIN EARLY STOPPING (API NUEVA)
                if name == 'XGBoost':
                    # Para XGBoost, usar GridSearchCV estándar sin early stopping (evitar errores de API)
                    print(f"🔍 Optimización estándar para {name} (GridSearchCV sin early stopping)...")

                    grid_search = GridSearchCV(
                        base_model.model,
                        model_config['param_grid'],
                        cv=cv_splitter,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    combinations = self._get_param_combinations(model_config['param_grid'])
                    print(f"🔍 Calculando {combinations} combinaciones de hiperparámetros...")

                    # Aplicar sample_weights si están disponibles
                    if sample_weights is not None:
                        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        grid_search.fit(X_train, y_train)

                    self._show_hyperparameter_search_details(grid_search, name)

                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    print(f"✅ Mejor CV score: {-grid_search.best_score_:.4f}")
                    
                else:
                    # Para otros modelos, usar GridSearchCV estándar
                    grid_search = GridSearchCV(
                        base_model.model,
                        model_config['param_grid'],
                        cv=cv_splitter,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )

                    print(f"🔍 DEBUG: param_grid = {model_config['param_grid']}")
                    combinations = self._get_param_combinations(model_config['param_grid'])
                    print(f"🔍 Calculando {combinations} combinaciones de hiperparámetros...")
                    
                    # Aplicar sample_weights si están disponibles
                    if sample_weights is not None:
                        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        grid_search.fit(X_train, y_train)
                    
                    # 📊 MOSTRAR DETALLES DE LA BÚSQUEDA DE HIPERPARÁMETROS
                    self._show_hyperparameter_search_details(grid_search, name)
                    
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_

                print(f"✅ Mejores parámetros: {best_params}")
                if name != 'XGBoost':
                    print(f"✅ Mejor CV score: {-grid_search.best_score_:.4f}")
            else:
                print(f"⚠️  Dataset pequeño ({len(X_train)} muestras), usando parámetros por defecto")
                base_model.train(X_train, y_train)
                best_model = base_model.model
                best_params = "default"

            print(f"📊 Evaluando con Cross-Validation...")
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=cv_splitter, 
                scoring='neg_mean_squared_error'
            )

            cv_mse_scores = -cv_scores
            cv_mean = cv_mse_scores.mean()
            cv_std = cv_mse_scores.std()

            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)


            tau_train, _ = kendalltau(y_train, y_pred_train)
            tau_test, _ = kendalltau(y_test, y_pred_test)

            rho_train, _ = spearmanr(y_train, y_pred_train)
            rho_test, _ = spearmanr(y_test, y_pred_test)

            metrics = {
                'cv_mse_mean': cv_mean,
                'cv_mse_std': cv_std,
                'cv_scores': cv_mse_scores.tolist(),
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'test_r2': r2_score(y_test, y_pred_test),
                'kendall_tau_train': tau_train,
                'kendall_tau_test': tau_test,
                'spearman_rho_train': rho_train,
                'spearman_rho_test': rho_test,
                'best_params': best_params,
                'overfitting_score': self._calculate_overfitting_score(
                    mean_squared_error(y_train, y_pred_train),
                    mean_squared_error(y_test, y_pred_test)
                ),
                # Nuevas métricas anti-overfitting
                'generalization_gap': mean_squared_error(y_test, y_pred_test) - mean_squared_error(y_train, y_pred_train),
                'cv_stability': cv_std / cv_mean if cv_mean > 0 else float('inf'),
                'train_test_r2_gap': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test)
            }

            # 🚨 DETECCIÓN AVANZADA DE OVERFITTING
            self._detect_overfitting_patterns(metrics, name)

            if metrics['train_mse'] < 0.0001 and metrics['test_mse'] > 0.05:
                print("🚨 El modelo está memorizando los datos de entrenamiento")

            self.results[name] = metrics
            self._show_model_results(name, metrics)

            optimized_model = model_class()
            optimized_model.model = best_model
            self._save_model(name, optimized_model)

        except Exception as e:
            print(f"❌ Error entrenando {name}: {e}")
            self.results[name] = {'error': str(e)}

    def _detect_overfitting_patterns(self, metrics, model_name):
        """Detecta patrones específicos de overfitting"""
        print(f"\n🔍 ANÁLISIS DE OVERFITTING - {model_name}:")
        
        # 1. Gap de generalización
        gen_gap = metrics['generalization_gap']
        if gen_gap > 5.0:
            print(f"   🚨 Gap de generalización alto: {gen_gap:.3f} (Train vs Test)")
        elif gen_gap > 2.0:
            print(f"   ⚠️  Gap de generalización moderado: {gen_gap:.3f}")
        else:
            print(f"   ✅ Gap de generalización bueno: {gen_gap:.3f}")
        
        # 2. Estabilidad del CV
        cv_stability = metrics['cv_stability']
        if cv_stability > 0.15:
            print(f"   🚨 CV inestable: {cv_stability:.3f} (alta variabilidad)")
        elif cv_stability > 0.10:
            print(f"   ⚠️  CV moderadamente estable: {cv_stability:.3f}")
        else:
            print(f"   ✅ CV estable: {cv_stability:.3f}")
        
        # 3. Gap de R²
        r2_gap = metrics['train_test_r2_gap']
        if r2_gap > 0.15:
            print(f"   🚨 R² gap alto: {r2_gap:.3f} (sobreajuste)")
        elif r2_gap > 0.08:
            print(f"   ⚠️  R² gap moderado: {r2_gap:.3f}")
        else:
            print(f"   ✅ R² gap bueno: {r2_gap:.3f}")
        
        # 4. Score overfitting combinado
        overfitting_level = self._calculate_overfitting_level(metrics)
        print(f"   📊 Nivel de overfitting: {overfitting_level}")
        
        # 5. Recomendaciones
        if any([gen_gap > 3.0, cv_stability > 0.12, r2_gap > 0.10]):
            print(f"   💡 RECOMENDACIONES:")
            if gen_gap > 3.0:
                print(f"      - Incrementar regularización")
                print(f"      - Reducir complejidad del modelo")
            if cv_stability > 0.12:
                print(f"      - Aumentar datos de entrenamiento")
                print(f"      - Usar ensemble methods")
            if r2_gap > 0.10:
                print(f"      - Feature selection más agresiva")
                print(f"      - Early stopping más temprano")

    def _calculate_overfitting_level(self, metrics):
        """Calcula un nivel combinado de overfitting"""
        weights = {
            'overfitting_score': 0.4,
            'generalization_gap': 0.3,
            'cv_stability': 0.2,
            'train_test_r2_gap': 0.1
        }
        
        # Normalizar métricas
        norm_overfitting = min(metrics['overfitting_score'] - 1.0, 1.0)  # 0-1 scale
        norm_gen_gap = min(metrics['generalization_gap'] / 10.0, 1.0)    # 0-1 scale
        norm_cv_stability = min(metrics['cv_stability'] / 0.2, 1.0)      # 0-1 scale
        norm_r2_gap = min(metrics['train_test_r2_gap'] / 0.2, 1.0)       # 0-1 scale
        
        combined_score = (
            weights['overfitting_score'] * norm_overfitting +
            weights['generalization_gap'] * norm_gen_gap +
            weights['cv_stability'] * norm_cv_stability +
            weights['train_test_r2_gap'] * norm_r2_gap
        )
        
        if combined_score < 0.3:
            return "🟢 Bajo (Bueno)"
        elif combined_score < 0.6:
            return "🟡 Moderado"
        else:
            return "🔴 Alto (Crítico)"

    def _get_param_combinations(self, param_grid):
        """Calcula el número total de combinaciones de hiperparámetros"""
        total = 1
        for param_values in param_grid.values():
            if isinstance(param_values, (list, tuple)):
                total *= len(param_values)
            else:
                total *= 1  # Si es un valor único
        return total

    def _show_hyperparameter_search_details(self, grid_search, model_name):
        """Muestra detalles detallados de la búsqueda de hiperparámetros"""
        
        # Obtener todos los resultados
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Mostrar top 5 mejores combinaciones
        top_results = results_df.nlargest(5, 'mean_test_score')
        
        print(f"🏆 TOP 5 MEJORES COMBINACIONES:")
        for i, (idx, row) in enumerate(top_results.iterrows(), 1):
            score = -row['mean_test_score']  # Convertir de negativo
            std = row['std_test_score']
            params = row['params']
            
            print(f"\n   {i}. Score: {score:.4f} (±{std:.4f})")
            print(f"      Parámetros: {params}")
        
        # Mostrar estadísticas generales
        all_scores = -results_df['mean_test_score']
        print(f"\n📊 ESTADÍSTICAS DE LA BÚSQUEDA:")
        print(f"   🎯 Mejor score: {all_scores.min():.4f}")
        print(f"   📈 Score promedio: {all_scores.mean():.4f}")
        print(f"   📉 Peor score: {all_scores.max():.4f}")
        print(f"   📏 Desviación estándar: {all_scores.std():.4f}")
        print(f"   🔄 Mejora vs promedio: {((all_scores.mean() - all_scores.min()) / all_scores.mean() * 100):.1f}%")
        
        # Analizar importancia de parámetros
        self._analyze_parameter_importance(results_df, model_name)

    def _analyze_parameter_importance(self, results_df, model_name):
        """Analiza qué parámetros tienen más impacto en el rendimiento"""
        print(f"\n🔍 ANÁLISIS DE IMPORTANCIA DE PARÁMETROS:")
        
        # Convertir params a columnas separadas
        param_columns = {}
        for idx, row in results_df.iterrows():
            for param, value in row['params'].items():
                if param not in param_columns:
                    param_columns[param] = []
                param_columns[param].append(value)
        
        scores = -results_df['mean_test_score']  # Convertir de negativo
        
        # Analizar cada parámetro
        for param_name, param_values in param_columns.items():
            unique_values = list(set(param_values))
            if len(unique_values) > 1:  # Solo analizar si hay variación
                print(f"\n   📊 {param_name}:")
                
                param_scores = {}
                for value in unique_values:
                    mask = [v == value for v in param_values]
                    value_scores = scores[mask]
                    if len(value_scores) > 0:
                        param_scores[value] = {
                            'mean': value_scores.mean(),
                            'std': value_scores.std(),
                            'count': len(value_scores)
                        }
                
                # Ordenar por score promedio
                sorted_params = sorted(param_scores.items(), key=lambda x: x[1]['mean'])
                
                for value, stats in sorted_params:
                    print(f"      {value}: {stats['mean']:.4f} (±{stats['std']:.4f}) [{stats['count']} pruebas]")
                
                # Mostrar el mejor valor para este parámetro
                best_value = sorted_params[0][0]
                worst_value = sorted_params[-1][0]
                improvement = sorted_params[-1][1]['mean'] - sorted_params[0][1]['mean']
                print(f"      ✅ Mejor: {best_value} | ❌ Peor: {worst_value} | 📈 Diferencia: {improvement:.4f}")

    def _calculate_overfitting_score(self, train_mse, test_mse):
        if train_mse == 0:
            return float('inf') if test_mse > 0 else 0
        return test_mse / train_mse

    def _show_model_results(self, name, metrics):
        if 'error' in metrics:
            print(f"\n❌ {name}: Error - {metrics['error']}")
            return

        print(f"\n📊 RESULTADOS DETALLADOS - {name}")
        print("-" * 40)
        print(f"🔄 Cross-Validation MSE: {metrics['cv_mse_mean']:.4f} ± {metrics['cv_mse_std']:.4f}")
        print(f"🏋️  Train MSE: {metrics['train_mse']:.4f}")
        print(f"🎯 Test MSE:  {metrics['test_mse']:.4f}")
        print(f"📈 Test R²:   {metrics['test_r2']:.4f}")
        print(f"🔗 Kendall’s tau (train): {metrics['kendall_tau_train']:.3f}")
        print(f"🔗 Kendall’s tau (test):  {metrics['kendall_tau_test']:.3f}")
        print(f"🔗 Spearman rho (test):   {metrics['spearman_rho_test']:.3f}")

        overfitting = metrics['overfitting_score']
        if overfitting < 1.1:
            print(f"✅ Overfitting Score: {overfitting:.2f} (Bueno)")
        elif overfitting < 1.5:
            print(f"⚠️  Overfitting Score: {overfitting:.2f} (Moderado)")
        else:
            print(f"🚨 Overfitting Score: {overfitting:.2f} (Alto)")

        if metrics['best_params'] != "default":
            print(f"🔧 Mejores parámetros: {metrics['best_params']}")

    def _show_detailed_comparison(self):
        print(f"\n{'='*80}")
        print("🏆 COMPARACIÓN DETALLADA DE MODELOS CON CROSS-VALIDATION")
        print(f"{'='*80}")

        print(f"{'Modelo':<15} {'CV MSE':<12} {'Test MSE':<10} {'Test R²':<8} {'Kendalls Tau':<12} {'Overfitting':<12} {'Estado'}")
        print("-" * 80)

        best_cv_score = float('inf')
        best_model = None

        for name, metrics in self.results.items():
            if 'error' in metrics:
                print(f"{name:<15} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8} {'ERROR':<12} {'❌'}")
                continue

            cv_score = metrics['cv_mse_mean']
            test_mse = metrics['test_mse']
            test_r2 = metrics['test_r2']
            overfitting = metrics['overfitting_score']
            kendall_tau = metrics['kendall_tau_test']

            if overfitting < 1.1:
                status = "✅ Bueno"
            elif overfitting < 1.5:
                status = "⚠️  Moderado"
            else:
                status = "🚨 Overfitting"

            print(f"{name:<15} {cv_score:<12.4f} {test_mse:<10.4f} "
                  f"{test_r2:<8.4f} {kendall_tau:<12.2f} {overfitting:<12.2f} {status}")

            if cv_score < best_cv_score and overfitting < 1.3:
                best_cv_score = cv_score
                best_model = name

        if best_model:
            print(f"\n🏆 MEJOR MODELO: {best_model}")
            print(f"   📊 CV MSE: {best_cv_score:.4f}")
            print(f"   🎯 Sin overfitting significativo")
        else:
            print(f"\n⚠️  ADVERTENCIA: Todos los modelos muestran problemas de overfitting")
            print(f"   💡 Considera reducir la complejidad o conseguir más datos")

    def _save_model(self, name, model):
        model_path = f"app/models_cache/{name.lower()}_model.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"💾 Modelo optimizado guardado: {model_path}")
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")

    def _save_metadata(self, label_encoder, feature_names):
        print("💾 Guardando metadata del modelo!!!!!!!!!!!!!...")
        print(f"   - Features: {len(feature_names) if feature_names else 0}")
        print(f"   - Label Encoder: {type(label_encoder).__name__ if label_encoder else 'None'}")
        print(f"RESULTADOS: {self.results}")
        try:
            with open("app/models_cache/label_encoder.pkl", 'wb') as f:
                pickle.dump(label_encoder, f)
            with open("app/models_cache/feature_names.pkl", 'wb') as f:
                pickle.dump(feature_names, f)
            print("✅ Metadata guardada")
        except Exception as e:
            print(f"❌ Error guardando metadata: {e}")
        

    def _save_training_results(self, results,label_encoder, feature_names):
        """Guarda métricas de entrenamiento para selección posterior"""
        try:
            results_file = "app/models_cache/training_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"💾 Métricas de entrenamiento guardadas: {results_file}")


            try:
                if self.results:
                    self._save_inference_manifest(feature_names, self.results, selection_metric="kendall")
            except Exception as e:
                print(f"⚠️ No se pudo guardar manifiesto de inferencia: {e}")
            
            # También guardar un resumen legible
            self._save_readable_summary(results)
            
        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")

    def _save_readable_summary(self, results):
        """Guarda un resumen legible de los resultados"""
        try:
            summary_file = "app/models_cache/model_comparison.txt"
            
            with open(summary_file, 'w') as f:
                f.write("COMPARACIÓN DE MODELOS - MÉTRICAS DE ENTRENAMIENTO\n")
                f.write("="*60 + "\n\n")
                
                for name, metrics in results.items():
                    if 'error' in metrics:
                        f.write(f"{name}: ERROR - {metrics['error']}\n\n")
                        continue
                    
                    f.write(f"{name}:\n")
                    f.write(f"  CV MSE: {metrics.get('cv_mse_mean', 'N/A'):.4f}\n")
                    f.write(f"  Test MSE: {metrics.get('test_mse', 'N/A'):.4f}\n")
                    f.write(f"  Test R²: {metrics.get('test_r2', 'N/A'):.4f}\n")
                    f.write(f"  Overfitting Score: {metrics.get('overfitting_score', 'N/A'):.2f}\n")
                    f.write(f"  Mejor Parámetros: {metrics.get('best_params', 'N/A')}\n\n")
            
            print(f"📄 Resumen legible guardado: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error guardando resumen: {e}")

    def calculate_year_weights(self, df_with_years):
        """
        Calcula pesos para las muestras basándose en el año
        Datos más recientes tienen mayor peso
        """
        if 'year' not in df_with_years.columns:
            print("⚠️ No se encontró columna 'year', usando pesos uniformes")
            return np.ones(len(df_with_years))
        
        # Mapeo de años a pesos según configuración
        year_weights_map = {
            2025: DATA_IMPORTANCE.get("2025_weight", 0.50),
            2024: DATA_IMPORTANCE.get("2024_weight", 0.25), 
            2023: DATA_IMPORTANCE.get("2023_weight", 0.15),
            2022: DATA_IMPORTANCE.get("2022_weight", 0.10)
        }
        
        # Calcular pesos para cada fila
        weights = df_with_years['year'].map(year_weights_map).fillna(0.05)  # peso mínimo para años no definidos
        
        # Normalizar para que la suma sea igual al número de muestras
        weights = weights * len(weights) / weights.sum()
        
        print(f"📅 Pesos aplicados por año:")
        year_counts = df_with_years['year'].value_counts().sort_index()
        for year in sorted(year_counts.index):
            weight = year_weights_map.get(year, 0.05)
            count = year_counts[year]
            print(f"   {year}: {weight:.1%} peso × {count} muestras")
        
        return weights.values

    # =======================
    # Feature audit utilities
    # =======================
    def _audit_features(self, feature_names, target_name=None, prefix="Entrenamiento"):
        if not feature_names:
            print("⚠️  Sin feature_names para auditar")
            return
        print(f"\n🧾 {prefix}: FEATURES USADAS ({len(feature_names)})")
        if target_name:
            print(f"   🎯 Target: {target_name}")
        safe, unsafe = [], []
        for f in feature_names:
            is_safe, reason = self._is_pre_race_safe(f)
            (safe if is_safe else unsafe).append((f, reason))
        if unsafe:
            print("   🚫 Sospecha post-race (revisar):")
            for name, reason in unsafe:
                print(f"      - {name}  [{reason}]")
        print("   ✅ Pre-race (ok):")
        for name, _ in safe:
            print(f"      - {name}")
        # Guardar auditoría legible
        try:
            out_path = "app/models_cache/feature_audit.txt"
            rows = [f"Target: {target_name}\n\n", "UNSAFE (post-race sospechoso):\n"]
            rows += [f"- {n} [{r}]\n" for n, r in unsafe]
            rows += ["\nSAFE (pre-race):\n"]
            rows += [f"- {n}\n" for n, _ in safe]
            with open(out_path, 'w') as f:
                f.writelines(rows)
            print(f"📄 Auditoría de features guardada: {out_path}")
        except Exception:
            pass

    def _is_pre_race_safe(self, name: str):
        n = (name or '').lower()
        # Señales seguramente post-race o que dependen del resultado del GP
        unsafe_tokens = [
            'race_position', 'final_position', 'grid_position', 'grid_to_race_change',
            'quali_vs_race_delta', 'points_efficiency', 'fastest_lap', 'status',
            'race_best_lap', 'lap_time_std', 'lap_time_consistency'
        ]
        # Evitar falso positivo: permitir fp3_best_time (sesión previa)
        if n == 'fp3_best_time':
            return True, 'fp3 (pre-quali)'
        if any(tok in n for tok in unsafe_tokens):
            return False, 'contiene métrica de carrera/resultado'
        # Candidatas pre-race (prácticas, clima, histórico, codificación)
        safe_prefixes = [
            'fp1_', 'fp2_', 'fp3_', 'session_', 'weather_', 'team_avg_position_',
            'driver_', 'team_', 'expected_grid_position', 'points_last_3',
            'avg_position_last_3', 'avg_quali_last_3', 'overtaking_ability',
            'team_track_avg_position', 'driver_track_avg_position', 'sector_',
            'heat_index', 'temp_deviation_from_ideal', 'weather_difficulty_index',
            'total_laps'
        ]
        if any(n.startswith(p) for p in safe_prefixes):
            return True, 'histórico/prácticas/clima'
        # Default: considerarla pre-race pero marcar como "revisar" si suena a quali directa
        if 'quali_position' in n:
            return False, 'quali directa (del evento)'
        return True, 'genérica'