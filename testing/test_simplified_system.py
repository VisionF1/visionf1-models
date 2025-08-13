"""
Test del sistema simplificado con valores por defecto
"""

import pandas as pd
import numpy as np
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

def test_simplified_system():
    """Prueba el sistema simplificado con valores por defecto"""
    
    print("🧪 TEST SISTEMA SIMPLIFICADO CON VALORES POR DEFECTO")
    print("=" * 55)
    
    # Datos de prueba con diferentes escenarios
    data = []
    
    # Veterano con historial completo
    for race_num in range(1, 6):
        data.append({
            'driver': 'Hamilton',
            'team': 'Mercedes',
            'race_name': f'Race_{race_num}',
            'year': 2024,
            'race_position': [2, 3, 1, 4, 3][race_num-1],
            'quali_position': [3, 2, 1, 5, 4][race_num-1],
            'points': [18, 15, 25, 12, 15][race_num-1]
        })
    
    # Piloto sin historial suficiente (solo aparece en carrera 1)
    data.append({
        'driver': 'Rookie',
        'team': 'Williams',
        'race_name': 'Race_1',
        'year': 2024,
        'race_position': 18,
        'quali_position': 19,
        'points': 0
    })
    
    # Piloto con poquito historial (aparece en carreras 4 y 5)
    for race_num in [4, 5]:
        data.append({
            'driver': 'NewDriver',
            'team': 'Alpine',
            'race_name': f'Race_{race_num}',
            'year': 2024,
            'race_position': [16, 14][race_num-4],
            'quali_position': [17, 15][race_num-4],
            'points': [0, 2][race_num-4]
        })
    
    df = pd.DataFrame(data)
    
    print("📊 DATOS DE PRUEBA:")
    for driver in df['driver'].unique():
        driver_data = df[df['driver'] == driver]
        carreras = len(driver_data)
        print(f"   {driver}: {carreras} carreras")
    
    # Aplicar feature engineering
    engineer = AdvancedFeatureEngineer()
    
    print(f"\n🔧 Aplicando features de momentum simplificadas...")
    df_enhanced = engineer.create_momentum_features(df.copy())
    
    # Analizar resultados por piloto
    print(f"\n📊 ANÁLISIS DE RESULTADOS:")
    print("=" * 35)
    
    for driver in df['driver'].unique():
        driver_data = df_enhanced[df_enhanced['driver'] == driver]
        print(f"\n🏁 {driver}:")
        print(f"   Carreras: {len(driver_data)}")
        
        # Mostrar features de momentum para cada carrera
        for idx, row in driver_data.iterrows():
            race = row['race_name']
            avg_pos = row['avg_position_last_3']
            points = row['points_last_3']
            trend = row['position_trend_last_5']
            
            print(f"   {race}:")
            print(f"      avg_position_last_3: {avg_pos:.1f}")
            print(f"      points_last_3: {points:.1f}")
            print(f"      position_trend_last_5: {trend:.2f}")
    
    # Verificar que no hay NaN
    print(f"\n🔍 VERIFICACIÓN DE COMPLETITUD:")
    print("=" * 35)
    
    momentum_features = ['avg_position_last_3', 'avg_quali_last_3', 'points_last_3', 'position_trend_last_5']
    
    for feature in momentum_features:
        if feature in df_enhanced.columns:
            nan_count = df_enhanced[feature].isna().sum()
            total = len(df_enhanced)
            completitud = ((total - nan_count) / total) * 100
            status = "✅" if nan_count == 0 else "❌"
            print(f"   {status} {feature}: {completitud:.1f}% completo")
    
    # Mostrar valores por defecto aplicados
    print(f"\n📋 VALORES POR DEFECTO UTILIZADOS:")
    print("=" * 40)
    print(f"   avg_position_last_3/5: 10.5 (posición media del grid)")
    print(f"   avg_quali_last_3: 10.5 (posición media de clasificación)")
    print(f"   points_last_3/5: 0 (sin puntos previos)")
    print(f"   position_trend_last_5: 0 (tendencia neutral)")

if __name__ == "__main__":
    test_simplified_system()
