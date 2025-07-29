import pickle
import os
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
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4]
                }
            },
            'XGBoost': {
                'model_class': XGBoostPredictor,
                'param_grid': {
                    'n_estimators': [30, 50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.6, 0.8],
                    'reg_alpha': [0.01, 0.1],  # L1 regularization
                    'reg_lambda': [0.01, 0.1]  # L2 regularization
                }
            },
            'GradientBoosting': {
                'model_class': GradientBoostingPredictor,
                'param_grid': {
                    'n_estimators': [30, 50],  # REDUCIR para evitar overfitting
                    'max_depth': [2, 3],       # REDUCIR profundidad
                    'learning_rate': [0.01, 0.05],  # LEARNING RATE más bajo
                    'subsample': [0.6, 0.8],
                    'min_samples_split': [10, 20],  # AUMENTAR splits mínimos
                    'min_samples_leaf': [5, 10]     # AUMENTAR hojas mínimas
                }
            }
        }

    def train_all_models(self, X_train, X_test, y_train, y_test, label_encoder, feature_names):
        print("\U0001F682 Entrenando modelos con Cross-Validation...")

        if X_train is None or y_train is None:
            print("❌ No hay datos para entrenar")
            return {}

        if len(X_train) < 30:
            print("🛑 Dataset chico: reduciendo hiperparámetros para evitar overfitting")
            self.models['RandomForest']['param_grid']['max_depth'] = [3]
            self.models['XGBoost']['param_grid']['max_depth'] = [3]
            self.models['GradientBoosting']['param_grid']['max_depth'] = [3]

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

            fit_params = {}
            if name == 'XGBoost':
                # Para XGBoost usar eval_set sin early_stopping en GridSearchCV
                fit_params = {}  # Vacío para GridSearchCV
            elif hasattr(base_model.model, "early_stopping_rounds"):
                fit_params = {
                    "eval_set": [(X_test, y_test)],
                    "early_stopping_rounds": 10,
                    "verbose": False
                }

            if len(X_train) > 20:
                # CONFIGURACIÓN ESPECIAL PARA XGBOOST
                if name == 'XGBoost':
                    grid_search = GridSearchCV(
                        base_model.model,
                        model_config['param_grid'],
                        cv=cv_splitter,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                        )
                else:
                    grid_search = GridSearchCV(
                        base_model.model,
                        model_config['param_grid'],
                        cv=cv_splitter,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )

                grid_search.fit(X_train, y_train)  # Sin fit_params para XGBoost
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                print(f"✅ Mejores parámetros: {best_params}")
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
                )
            }

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