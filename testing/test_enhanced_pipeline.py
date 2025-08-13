"""
Test completo del pipeline mejorado con features avanzadas
Ejecuta un entrenamiento completo comparando pipeline básico vs avanzado
"""

import pandas as pd
import numpy as np
from app.core.pipeline import Pipeline
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
from app.core.training.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def compare_pipelines():
    print("🏎️ COMPARACIÓN PIPELINE BÁSICO VS AVANZADO")
    print("=" * 60)
    
    # Cargar datos reales
    print("📂 Cargando datos...")
    df = pd.read_pickle('app/models_cache/cached_data.pkl')
    print(f"   ✅ Dataset cargado: {df.shape}")
    
    # === PIPELINE BÁSICO (sin features avanzadas) ===
    print(f"\n🔷 PIPELINE BÁSICO (sin features avanzadas)")
    print("=" * 60)
    
    preparer_basic = EnhancedDataPreparer(use_advanced_features=False)
    X_basic, y_basic, encoder_basic, features_basic = preparer_basic.prepare_enhanced_features(df.copy())
    
    if X_basic is not None:
        print(f"   📊 Features básicas: {X_basic.shape[1]}")
        print(f"   📊 Registros: {X_basic.shape[0]}")
        
        # Train/test split
        X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(
            X_basic, y_basic, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo básico
        trainer_basic = ModelTrainer(use_time_series_cv=True)
        results_basic = trainer_basic.train_all_models(
            X_train_basic, X_test_basic, y_train_basic, y_test_basic,
            encoder_basic, features_basic
        )
    
    # === PIPELINE AVANZADO (con features avanzadas) ===
    print(f"\n🚀 PIPELINE AVANZADO (con features avanzadas)")
    print("=" * 60)
    
    preparer_advanced = EnhancedDataPreparer(use_advanced_features=True)
    X_advanced, y_advanced, encoder_advanced, features_advanced = preparer_advanced.prepare_enhanced_features(df.copy())
    
    if X_advanced is not None:
        print(f"   📊 Features avanzadas: {X_advanced.shape[1]}")
        print(f"   📊 Registros: {X_advanced.shape[0]}")
        
        # Train/test split
        X_train_advanced, X_test_advanced, y_train_advanced, y_test_advanced = train_test_split(
            X_advanced, y_advanced, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo avanzado
        trainer_advanced = ModelTrainer(use_time_series_cv=True)
        results_advanced = trainer_advanced.train_all_models(
            X_train_advanced, X_test_advanced, y_train_advanced, y_test_advanced,
            encoder_advanced, features_advanced
        )
    
    # === COMPARACIÓN DE RESULTADOS ===
    print(f"\n📊 COMPARACIÓN DE RESULTADOS")
    print("=" * 60)
    
    print(f"{'Modelo':<20} | {'Pipeline':<10} | {'CV MSE':<10} | {'Test MSE':<10} | {'R²':<10}")
    print("-" * 75)
    
    # Resultados básicos
    if 'results_basic' in locals():
        for model_name, metrics in results_basic.items():
            if 'error' not in metrics:
                cv_mse = metrics.get('cv_mse', 'N/A')
                test_mse = metrics.get('test_mse', 'N/A')
                test_r2 = metrics.get('test_r2', 'N/A')
                
                cv_str = f"{cv_mse:.2f}" if isinstance(cv_mse, (int, float)) else str(cv_mse)
                test_str = f"{test_mse:.2f}" if isinstance(test_mse, (int, float)) else str(test_mse)
                r2_str = f"{test_r2:.3f}" if isinstance(test_r2, (int, float)) else str(test_r2)
                
                print(f"{model_name:<20} | {'Básico':<10} | {cv_str:<10} | {test_str:<10} | {r2_str:<10}")
    
    # Resultados avanzados
    if 'results_advanced' in locals():
        for model_name, metrics in results_advanced.items():
            if 'error' not in metrics:
                cv_mse = metrics.get('cv_mse', 'N/A')
                test_mse = metrics.get('test_mse', 'N/A')
                test_r2 = metrics.get('test_r2', 'N/A')
                
                cv_str = f"{cv_mse:.2f}" if isinstance(cv_mse, (int, float)) else str(cv_mse)
                test_str = f"{test_mse:.2f}" if isinstance(test_mse, (int, float)) else str(test_mse)
                r2_str = f"{test_r2:.3f}" if isinstance(test_r2, (int, float)) else str(test_r2)
                
                print(f"{model_name:<20} | {'Avanzado':<10} | {cv_str:<10} | {test_str:<10} | {r2_str:<10}")
    
    # === ANÁLISIS DE MEJORA ===
    print(f"\n🎯 ANÁLISIS DE MEJORA")
    print("=" * 60)
    
    if 'results_basic' in locals() and 'results_advanced' in locals():
        for model_name in ['RandomForest', 'XGBoost', 'GradientBoosting']:
            if (model_name in results_basic and model_name in results_advanced and 
                'error' not in results_basic[model_name] and 'error' not in results_advanced[model_name]):
                
                basic_mse = results_basic[model_name].get('cv_mse')
                advanced_mse = results_advanced[model_name].get('cv_mse')
                
                if basic_mse and advanced_mse and isinstance(basic_mse, (int, float)) and isinstance(advanced_mse, (int, float)):
                    improvement = ((basic_mse - advanced_mse) / basic_mse) * 100
                    
                    print(f"📈 {model_name}:")
                    print(f"   MSE Básico: {basic_mse:.2f}")
                    print(f"   MSE Avanzado: {advanced_mse:.2f}")
                    print(f"   Mejora: {improvement:.1f}%")
    
    # === ANÁLISIS DE FEATURES ===
    print(f"\n🔍 ANÁLISIS DE FEATURES")
    print("=" * 60)
    
    if 'features_basic' in locals() and 'features_advanced' in locals():
        print(f"Features básicas: {len(features_basic)}")
        print(f"Features avanzadas: {len(features_advanced)}")
        print(f"Incremento: {len(features_advanced) - len(features_basic)} features")
        
        # Features nuevas añadidas
        new_features = [f for f in features_advanced if f not in features_basic]
        if new_features:
            print(f"\n🆕 Features añadidas ({len(new_features)}):")
            for feature in new_features:
                print(f"   ✅ {feature}")
    
    print(f"\n🎉 COMPARACIÓN COMPLETADA!")

def test_robustness():
    """Test de robustez del pipeline mejorado"""
    print(f"\n🔧 TEST DE ROBUSTEZ DEL PIPELINE")
    print("=" * 60)
    
    # Test con diferentes configuraciones
    configurations = [
        {"use_advanced_features": False, "name": "Básico"},
        {"use_advanced_features": True, "name": "Avanzado"}
    ]
    
    for config in configurations:
        print(f"\n📋 Configuración: {config['name']}")
        print("-" * 30)
        
        try:
            preparer = EnhancedDataPreparer(use_advanced_features=config['use_advanced_features'])
            df = pd.read_pickle('app/models_cache/cached_data.pkl')
            
            X, y, encoder, features = preparer.prepare_enhanced_features(df.copy())
            
            if X is not None:
                print(f"   ✅ Preparación exitosa: {X.shape}")
                print(f"   📊 Features: {len(features)}")
                
                # Verificar que no hay NaN
                nan_count = X.isnull().sum().sum()
                print(f"   🧹 Valores NaN: {nan_count}")
                
                # Verificar tipos de datos
                numeric_cols = X.select_dtypes(include=[np.number]).shape[1]
                print(f"   🔢 Columnas numéricas: {numeric_cols}/{X.shape[1]}")
                
                if config['use_advanced_features']:
                    # Verificar features específicas que implementamos
                    implemented_features = [
                        'quali_gap_to_pole', 'team_quali_rank', 'weather_difficulty_index',
                        'heat_index', 'avg_position_last_3'
                    ]
                    
                    found_features = [f for f in implemented_features if f in features]
                    print(f"   🎯 Features implementadas encontradas: {len(found_features)}/{len(implemented_features)}")
                    for f in found_features:
                        print(f"      ✅ {f}")
                
            else:
                print(f"   ❌ Error en preparación de datos")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    compare_pipelines()
    test_robustness()
