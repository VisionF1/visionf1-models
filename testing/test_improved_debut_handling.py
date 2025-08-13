"""
Test completo del manejo mejorado de debuts absolutos
"""

import pandas as pd
import numpy as np
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

def test_improved_debut_handling():
    """Prueba el manejo mejorado de debuts absolutos"""
    
    print("🧪 TEST COMPLETO - DEBUTS ABSOLUTOS MEJORADOS")
    print("=" * 55)
    
    # Crear datos realistas con múltiples casos
    data = []
    
    # Veteranos con historial completo (para contexto y expectativas de equipo)
    for race_num in range(1, 6):  # 5 carreras
        for driver_info in [
            ('Hamilton', 'Mercedes', [2, 3, 1, 4, 3]),
            ('Verstappen', 'Red Bull Racing', [1, 1, 2, 1, 1]),
            ('Leclerc', 'Ferrari', [3, 4, 5, 3, 4]),
            ('Albon', 'Williams', [15, 14, 16, 13, 12])  # Contexto para Williams
        ]:
            driver, team, positions = driver_info
            data.append({
                'driver': driver,
                'race_name': f'Race_{race_num}',
                'year': 2024,
                'race_position': positions[race_num-1],
                'quali_position': positions[race_num-1] + np.random.randint(-1, 2),
                'points': max(0, 26 - positions[race_num-1]*2),
                'team': team
            })
    
    # CASOS DE DEBUT:
    
    # 1. Debut absoluto en carrera 1
    data.append({
        'driver': 'Colapinto_R1',  # Debut en carrera 1
        'race_name': 'Race_1',
        'year': 2024,
        'race_position': 18,
        'quali_position': 19,
        'points': 0,
        'team': 'Williams'
    })
    
    # 2. Debut absoluto en carrera 3 (sustituto)
    data.append({
        'driver': 'Doohan_R3',  # Debut en carrera 3
        'race_name': 'Race_3',
        'year': 2024,
        'race_position': 19,
        'quali_position': 20,
        'points': 0,
        'team': 'Alpine'
    })
    
    # 3. Debut con expectativas diferentes (equipo top)
    data.append({
        'driver': 'Antonelli_R5',  # Debut en Mercedes
        'race_name': 'Race_5',
        'year': 2024,
        'race_position': 8,  # Mejor resultado por mejor equipo
        'quali_position': 9,
        'points': 4,
        'team': 'Mercedes'
    })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['driver', 'race_name']).reset_index(drop=True)
    
    # Mostrar expectativas por equipo
    print("📊 EXPECTATIVAS POR EQUIPO:")
    team_expectations = df.groupby('team')['race_position'].median()
    for team, expectation in team_expectations.items():
        print(f"   {team}: posición {expectation:.1f}")
    
    # Aplicar feature engineering mejorado
    engineer = AdvancedFeatureEngineer()
    print(f"\n🔧 Aplicando feature engineering mejorado...")
    
    df_enhanced = engineer.create_momentum_features(df.copy())
    
    # Analizar cada debutante
    debuts = [name for name in df['driver'].unique() if any(x in name for x in ['Colapinto', 'Doohan', 'Antonelli'])]
    
    print(f"\n👶 ANÁLISIS DE DEBUTANTES:")
    print("=" * 40)
    
    for debut_driver in debuts:
        debut_data = df_enhanced[df_enhanced['driver'] == debut_driver]
        
        if len(debut_data) > 0:
            debut_row = debut_data.iloc[0]
            team = debut_row['team']
            team_expectation = team_expectations[team]
            
            print(f"\n🏁 {debut_driver}:")
            print(f"   Equipo: {team}")
            print(f"   Expectativa equipo: {team_expectation:.1f}")
            print(f"   Posición real: {debut_row['race_position']}")
            print(f"   Es debut: {'✅ Sí' if debut_row['is_debut'] else '❌ No'}")
            print(f"   avg_position_last_3: {debut_row['avg_position_last_3']:.1f}")
            print(f"   avg_quali_last_3: {debut_row['avg_quali_last_3']:.1f}")
            print(f"   points_last_3: {debut_row['points_last_3']:.1f}")
            print(f"   position_trend_last_5: {debut_row['position_trend_last_5']:.1f}")
    
    # Verificar que no hay NaN en features de momentum
    momentum_features = ['avg_position_last_3', 'avg_quali_last_3', 'points_last_3', 'position_trend_last_5']
    
    print(f"\n🔍 VERIFICACIÓN DE COMPLETITUD:")
    print("=" * 35)
    
    for feature in momentum_features:
        if feature in df_enhanced.columns:
            total_values = len(df_enhanced)
            nan_count = df_enhanced[feature].isna().sum()
            completitud = ((total_values - nan_count) / total_values) * 100
            status = "✅" if nan_count == 0 else "❌"
            print(f"   {status} {feature}: {completitud:.1f}% completo ({nan_count} NaN)")
    
    # Comparar debuts vs veteranos
    print(f"\n📊 COMPARACIÓN DEBUTS VS VETERANOS:")
    print("=" * 42)
    
    debuts_mask = df_enhanced['is_debut'] == True
    veterans_mask = df_enhanced['is_debut'] == False
    
    if debuts_mask.sum() > 0 and veterans_mask.sum() > 0:
        debuts_avg = df_enhanced[debuts_mask]['avg_position_last_3'].mean()
        veterans_avg = df_enhanced[veterans_mask]['avg_position_last_3'].mean()
        
        print(f"   Debuts - avg_position_last_3: {debuts_avg:.1f}")
        print(f"   Veteranos - avg_position_last_3: {veterans_avg:.1f}")
        print(f"   Diferencia: {abs(debuts_avg - veterans_avg):.1f} posiciones")

if __name__ == "__main__":
    test_improved_debut_handling()
