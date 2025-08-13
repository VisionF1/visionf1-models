#!/usr/bin/env python3
"""
Test del Feature Engineering Avanzado
Prueba las nuevas características y compara el rendimiento
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
from app.core.training.model_trainer import ModelTrainer
from app.data.collectors.fastf1_collector import FastF1Collector
import pandas as pd
import pickle

def test_advanced_features():
    print("🧪 TESTING FEATURE ENGINEERING AVANZADO")
    print("=" * 60)
    
    # 1. Cargar datos
    print("📦 Cargando datos desde cache...")
    cache_file = "app/models_cache/cached_data.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"   ✅ Datos cargados: {len(data)} registros")
    else:
        print("   ❌ No se encontró cache. Ejecuta primero main.py 1")
        return
    
    # 2. Probar preparación BÁSICA
    print(f"\n🔍 ENTRENAMIENTO CON FEATURES BÁSICAS")
    print("-" * 40)
    
    basic_preparer = EnhancedDataPreparer(use_advanced_features=False)
    X_basic, y_basic, label_encoder_basic, features_basic = basic_preparer.prepare_enhanced_features(data)
    
    if X_basic is not None:
        print(f"   📊 Features básicas: {X_basic.shape[1]}")
        
        # Entrenar modelo básico (solo RandomForest para rapidez)
        basic_trainer = ModelTrainer()
        basic_trainer.models = {'RandomForest': basic_trainer.models['RandomForest']}  # Solo RF
        
        from sklearn.model_selection import train_test_split
        X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(
            X_basic, y_basic, test_size=0.2, random_state=42
        )
        
        basic_results = basic_trainer.train_all_models(
            X_train_basic, X_test_basic, y_train_basic, y_test_basic, 
            label_encoder_basic, features_basic
        )
        
        basic_cv_score = basic_results.get('RandomForest', {}).get('cv_mse_mean', float('inf'))
        print(f"   🎯 CV MSE Básico: {basic_cv_score:.4f}")
    
    # 3. Probar preparación AVANZADA
    print(f"\n🚀 ENTRENAMIENTO CON FEATURES AVANZADAS")
    print("-" * 40)
    
    advanced_preparer = EnhancedDataPreparer(use_advanced_features=True)
    X_advanced, y_advanced, label_encoder_advanced, features_advanced = advanced_preparer.prepare_enhanced_features(data)
    
    if X_advanced is not None:
        print(f"   📊 Features avanzadas: {X_advanced.shape[1]}")
        print(f"   📈 Incremento: {X_advanced.shape[1] - X_basic.shape[1]} features adicionales")
        
        # Mostrar resumen de features
        print(advanced_preparer.get_feature_importance_summary())
        
        # Entrenar modelo avanzado (solo RandomForest para rapidez)
        advanced_trainer = ModelTrainer()
        advanced_trainer.models = {'RandomForest': advanced_trainer.models['RandomForest']}  # Solo RF
        
        X_train_advanced, X_test_advanced, y_train_advanced, y_test_advanced = train_test_split(
            X_advanced, y_advanced, test_size=0.2, random_state=42
        )
        
        advanced_results = advanced_trainer.train_all_models(
            X_train_advanced, X_test_advanced, y_train_advanced, y_test_advanced, 
            label_encoder_advanced, features_advanced
        )
        
        advanced_cv_score = advanced_results.get('RandomForest', {}).get('cv_mse_mean', float('inf'))
        print(f"   🎯 CV MSE Avanzado: {advanced_cv_score:.4f}")
    
    # 4. Comparación
    print(f"\n📊 COMPARACIÓN DE RESULTADOS")
    print("=" * 40)
    
    if X_basic is not None and X_advanced is not None:
        improvement = basic_cv_score - advanced_cv_score
        improvement_pct = (improvement / basic_cv_score) * 100 if basic_cv_score > 0 else 0
        
        print(f"🔍 Features básicas:")
        print(f"   📊 Cantidad: {X_basic.shape[1]} features")
        print(f"   🎯 CV MSE: {basic_cv_score:.4f}")
        
        print(f"\n🚀 Features avanzadas:")
        print(f"   📊 Cantidad: {X_advanced.shape[1]} features")
        print(f"   🎯 CV MSE: {advanced_cv_score:.4f}")
        
        print(f"\n📈 MEJORA:")
        if improvement > 0:
            print(f"   ✅ Mejora: {improvement:.4f} MSE ({improvement_pct:.2f}%)")
            print(f"   🏆 Las features avanzadas MEJORAN el rendimiento")
        elif improvement < 0:
            print(f"   ❌ Empeora: {abs(improvement):.4f} MSE ({abs(improvement_pct):.2f}%)")
            print(f"   ⚠️  Las features avanzadas pueden estar causando overfitting")
        else:
            print(f"   ➖ Sin cambio significativo")
        
        # Mostrar top features avanzadas creadas
        print(f"\n🎯 TOP FEATURES AVANZADAS CREADAS:")
        advanced_only = [f for f in features_advanced if f not in features_basic]
        for i, feature in enumerate(advanced_only[:10], 1):
            print(f"   {i:2d}. {feature}")
    
    print(f"\n🎉 TEST COMPLETADO")

if __name__ == "__main__":
    test_advanced_features()
