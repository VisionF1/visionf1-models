import pickle
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit, KFold
import numpy as np
from app.core.predictors.random_forest import RandomForestPredictor
from app.core.predictors.xgboost_model import XGBoostPredictor
from app.core.predictors.gradient_boosting import GradientBoostingPredictor

class ModelTrainer:
    def __init__(self, use_time_series_cv=True):
        self.use_time_series_cv = use_time_series_cv
        self.models = self._initialize_models_with_hyperparams()
        self.results = {}

    def _initialize_models_with_hyperparams(self):
        return {
            'RandomForest': {
                'model_class': RandomForestPredictor,
                'param_grid': {
                    'n_estimators': [50, 100],      # Solo 2 opciones
                    'max_depth': [3, 4],            # Solo 2 opciones
                    'min_samples_split': [10, 20],  # Solo 2 opciones
                    'min_samples_leaf': [5, 10],    # Solo 2 opciones
                    'max_features': ['sqrt', 0.5],  # Solo 2 opciones
                    'bootstrap': [True],            # Fijo
                    'oob_score': [True]             # Fijo - TOTAL: 2^5 = 32 combinaciones
                }
            },
            'XGBoost': {
                'model_class': XGBoostPredictor,
                'param_grid': {
                    'n_estimators': [50, 100],      # Solo 2 opciones
                    'max_depth': [3, 4],            # Solo 2 opciones  
                    'learning_rate': [0.05, 0.1],  # Solo 2 opciones
                    'subsample': [0.7, 0.8],       # Solo 2 opciones
                    'colsample_bytree': [0.7, 0.8], # Solo 2 opciones
                    'reg_alpha': [0.1, 0.5],       # Solo 2 opciones
                    'reg_lambda': [0.1, 0.5],      # Solo 2 opciones
                    'min_child_weight': [3, 5]     # Solo 2 opciones - TOTAL: 2^8 = 256 combinaciones
                }
            },
            'GradientBoosting': {
                'model_class': GradientBoostingPredictor,
                'param_grid': {
                    'n_estimators': [30, 50],       # Solo 2 opciones
                    'max_depth': [2, 3],            # Solo 2 opciones
                    'learning_rate': [0.05, 0.1],  # Solo 2 opciones
                    'subsample': [0.7, 0.8],       # Solo 2 opciones
                    'min_samples_split': [15, 25],  # Solo 2 opciones
                    'min_samples_leaf': [8, 12],    # Solo 2 opciones
                    'max_features': ['sqrt', 0.5],  # Solo 2 opciones
                    'validation_fraction': [0.1]    # Fijo - TOTAL: 2^7 = 128 combinaciones
                }
            }
        }

    def train_all_models(self, X_train, X_test, y_train, y_test, label_encoder, feature_names):
        print("\U0001F682 Entrenando modelos con Cross-Validation...")

        if X_train is None or y_train is None:
            print("❌ No hay datos para entrenar")
            return {}

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
        cv_splitter = self._get_cv_splitter(X_train)

        for name, model_config in self.models.items():
            print(f"\n{'='*50}")
            print(f"🔍 ENTRENANDO {name} CON CROSS-VALIDATION")
            print(f"{'='*50}")

            self._train_single_model_with_cv(
                name, model_config, X_train, X_test, y_train, y_test, cv_splitter
            )

        self._show_detailed_comparison()
        self._save_training_results(self.results)
        
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
                param_grid['max_depth'] = [2, 3]  # Limitar profundidad
            if 'n_estimators' in param_grid:
                param_grid['n_estimators'] = [20, 30, 50]  # Menos estimadores

    def _get_cv_splitter(self, X_train):
        n_samples = len(X_train)
        if self.use_time_series_cv and n_samples > 10:
            n_splits = min(5, n_samples // 5)
            print(f"📊 Usando TimeSeriesSplit con {n_splits} splits")
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            n_splits = min(3, n_samples // 5)
            print(f"📊 Usando KFold con {n_splits} splits")
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _train_single_model_with_cv(self, name, model_config, X_train, X_test, y_train, y_test, cv_splitter):
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
                    # Para XGBoost, usar GridSearchCV estándar sin early stopping
                    print(f"🔍 Optimización estándar para {name} (sin early stopping)...")
                    
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
                    grid_search.fit(X_train, y_train)
                    
                    # 📊 MOSTRAR DETALLES DE LA BÚSQUEDA DE HIPERPARÁMETROS
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
        print(f"\n🔬 ANÁLISIS DETALLADO DE HIPERPARÁMETROS - {model_name}")
        print("-" * 60)
        
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

        print(f"{'Modelo':<15} {'CV MSE':<12} {'Test MSE':<10} {'Test R²':<8} {'Overfitting':<12} {'Estado'}")
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

            if overfitting < 1.1:
                status = "✅ Bueno"
            elif overfitting < 1.5:
                status = "⚠️  Moderado"
            else:
                status = "🚨 Overfitting"

            print(f"{name:<15} {cv_score:<12.4f} {test_mse:<10.4f} "
                  f"{test_r2:<8.4f} {overfitting:<12.2f} {status}")

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
        try:
            with open("app/models_cache/label_encoder.pkl", 'wb') as f:
                pickle.dump(label_encoder, f)
            with open("app/models_cache/feature_names.pkl", 'wb') as f:
                pickle.dump(feature_names, f)
            print("✅ Metadata guardada")
        except Exception as e:
            print(f"❌ Error guardando metadata: {e}")

    def _save_training_results(self, results):
        """Guarda métricas de entrenamiento para selección posterior"""
        try:
            results_file = "app/models_cache/training_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"💾 Métricas de entrenamiento guardadas: {results_file}")
            
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