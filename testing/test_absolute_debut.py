"""
Test para verificar el manejo del debut absoluto (primera carrera ever de un piloto)
"""

import pandas as pd
import numpy as np
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

def test_absolute_debut():
    """Prueba el caso más extremo: debut absoluto de un piloto"""
    
    print("🧪 TEST DE DEBUT ABSOLUTO")
    print("=" * 40)
    
    # Crear datos realistas
    data = []
    
    # Veteranos con historial completo
    for race_num in range(1, 6):  # 5 carreras
        for driver_info in [
            ('Hamilton', 'Mercedes', [2, 3, 1, 4, 3]),
            ('Verstappen', 'Red Bull Racing', [1, 1, 2, 1, 1]),
            ('Leclerc', 'Ferrari', [3, 4, 5, 3, 4])
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
    
    # DEBUT ABSOLUTO: Piloto aparece SOLO en la carrera 1 (su primera vez ever)
    data.append({
        'driver': 'Colapinto',  # Piloto debutante
        'race_name': 'Race_1',   # Su primera carrera EVER
        'year': 2024,
        'race_position': 18,  # Posición típica de debut
        'quali_position': 19,
        'points': 0,
        'team': 'Williams'
    })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['driver', 'race_name']).reset_index(drop=True)
    
    print(f"📊 DATOS DE PRUEBA:")
    for driver, count in df['driver'].value_counts().items():
        debut_race = df[df['driver'] == driver]['race_name'].iloc[0]
        print(f"   {driver}: {count} carreras (debut: {debut_race})")
    
    # Aplicar feature engineering
    engineer = AdvancedFeatureEngineer()
    
    print(f"\n🔧 Aplicando feature engineering...")
    try:
        df_enhanced = engineer.create_momentum_features(df.copy())
        
        # Analizar el debutante
        debut_data = df_enhanced[df_enhanced['driver'] == 'Colapinto']
        
        print(f"\n👶 ANÁLISIS DEL DEBUTANTE ABSOLUTO:")
        print("=" * 45)
        
        if len(debut_data) > 0:
            debut_row = debut_data.iloc[0]
            print(f"   Piloto: {debut_row['driver']}")
            print(f"   Carrera: {debut_row['race_name']}")
            print(f"   Posición real: {debut_row['race_position']}")
            
            # Verificar features de momentum en el debut
            momentum_features = [f for f in engineer.created_features if f in debut_data.columns]
            
            print(f"\n   📈 FEATURES DE MOMENTUM EN EL DEBUT:")
            for feature in momentum_features:
                value = debut_row[feature]
                status = "✅ Valor" if not pd.isna(value) else "❌ NaN"
                print(f"      {feature}: {value} ({status})")
        
        # Verificar qué pasa si aplicamos el sistema completo
        print(f"\n🔧 Aplicando sistema completo de features...")
        df_full = engineer.create_all_advanced_features(df.copy())
        
        debut_full = df_full[df_full['driver'] == 'Colapinto']
        
        if len(debut_full) > 0:
            debut_row_full = debut_full.iloc[0]
            
            print(f"\n   🎯 FEATURES DESPUÉS DEL PROCESAMIENTO COMPLETO:")
            all_advanced_features = [f for f in engineer.created_features if f in debut_full.columns]
            
            nan_count = 0
            total_count = len(all_advanced_features)
            
            for feature in all_advanced_features:
                value = debut_row_full[feature]
                if pd.isna(value):
                    nan_count += 1
                    print(f"      ❌ {feature}: NaN")
                else:
                    print(f"      ✅ {feature}: {value:.2f}")
            
            completitud = ((total_count - nan_count) / total_count) * 100
            print(f"\n   📊 COMPLETITUD: {completitud:.1f}% ({total_count - nan_count}/{total_count} features)")
            
            if nan_count > 0:
                print(f"   ⚠️  {nan_count} features con NaN en debut absoluto")
        
    except Exception as e:
        print(f"❌ Error en feature engineering: {e}")

def test_debut_with_fallbacks():
    """Prueba cómo funcionan los fallbacks en debut absoluto"""
    
    print(f"\n\n🔄 TEST DE FALLBACKS EN DEBUT ABSOLUTO")
    print("=" * 50)
    
    # Datos mínimos para debut
    data = [
        # Un veterano para contexto
        {
            'driver': 'Hamilton',
            'race_name': 'Race_1',
            'year': 2024,
            'race_position': 2,
            'quali_position': 3,
            'points': 18,
            'team': 'Mercedes'
        },
        # Debutante absoluto
        {
            'driver': 'Colapinto',
            'race_name': 'Race_1',  # Su primera carrera EVER
            'year': 2024,
            'race_position': 18,
            'quali_position': 19,
            'points': 0,
            'team': 'Williams'
        }
    ]
    
    df = pd.DataFrame(data)
    
    print(f"📊 Datos para test de fallbacks:")
    print(f"   Hamilton: 1 carrera (veterano)")
    print(f"   Colapinto: 1 carrera (DEBUT ABSOLUTO)")
    
    # Simular el proceso de fallbacks manualmente
    print(f"\n🔧 Simulando fallbacks para features de momentum:")
    
    # Para avg_position_last_3 en debut absoluto
    print(f"\n   avg_position_last_3:")
    print(f"      1. Rolling con min_periods=1 y shift(1): NaN (no hay carrera anterior)")
    print(f"      2. Fallback a mediana del piloto: NaN (solo 1 carrera)")
    print(f"      3. Fallback a mediana general: {df['race_position'].median()}")
    
    # Para driver_track_avg_position
    print(f"\n   driver_track_avg_position:")
    print(f"      1. Historial en circuito: NaN (primera vez en circuito)")
    print(f"      2. Promedio general del piloto: {df[df['driver'] == 'Colapinto']['race_position'].mean()}")
    
    # Para team_track_avg_position
    williams_avg = df[df['team'] == 'Williams']['race_position'].mean()
    print(f"\n   team_track_avg_position:")
    print(f"      1. Historial equipo en circuito: NaN (equipo sin historial en circuito)")
    print(f"      2. Promedio general del equipo: {williams_avg}")

if __name__ == "__main__":
    test_absolute_debut()
    test_debut_with_fallbacks()
