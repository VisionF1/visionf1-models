"""
Test para verificar cómo se manejan los pilotos rookies sin historial
"""

import pandas as pd
import numpy as np
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

# Crear datos de ejemplo con un rookie
def create_rookie_test_data():
    """Crea datos de prueba con un piloto rookie"""
    
    # Datos de pilotos veteranos
    veteran_data = []
    for race_num in range(1, 6):  # 5 carreras
        for driver in ['Hamilton', 'Verstappen', 'Leclerc']:
            veteran_data.append({
                'driver': driver,
                'race_name': f'Race_{race_num}',
                'year': 2024,
                'race_position': np.random.randint(1, 10),
                'quali_position': np.random.randint(1, 10),
                'points': np.random.randint(0, 25),
                'team': 'Ferrari' if driver == 'Leclerc' else 'Mercedes' if driver == 'Hamilton' else 'Red Bull Racing'
            })
    
    # Datos del rookie (solo aparece en carreras 4 y 5)
    rookie_data = []
    for race_num in [4, 5]:
        rookie_data.append({
            'driver': 'Bearman',  # Rookie
            'race_name': f'Race_{race_num}',
            'year': 2024,
            'race_position': np.random.randint(12, 18),  # Posiciones típicas de rookie
            'quali_position': np.random.randint(15, 20),
            'points': 0,  # Típico para rookie
            'team': 'Ferrari'
        })
    
    df = pd.DataFrame(veteran_data + rookie_data)
    df = df.sort_values(['driver', 'race_name']).reset_index(drop=True)
    
    return df

def test_rookie_features():
    """Prueba cómo se generan features para un rookie"""
    
    print("🧪 PROBANDO MANEJO DE ROOKIES")
    print("=" * 50)
    
    # Crear datos de prueba
    df = create_rookie_test_data()
    
    print(f"📊 Datos originales:")
    print(f"   Carreras totales: {len(df)}")
    print(f"   Pilotos únicos: {df['driver'].nunique()}")
    print(f"   Carreras por piloto:")
    for driver, count in df['driver'].value_counts().items():
        print(f"      {driver}: {count} carreras")
    
    # Crear feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Aplicar feature engineering
    print(f"\n🔧 Aplicando feature engineering...")
    df_enhanced = engineer.create_momentum_features(df.copy())
    
    # Analizar features del rookie
    rookie_data = df_enhanced[df_enhanced['driver'] == 'Bearman'].copy()
    
    print(f"\n👨‍🚀 ANÁLISIS DEL ROOKIE (Bearman):")
    print("=" * 40)
    
    momentum_features = [f for f in engineer.created_features if f in rookie_data.columns and 'last_' in f]
    
    for idx, row in rookie_data.iterrows():
        race = row['race_name']
        print(f"\n🏁 {race}:")
        print(f"   Posición real: {row['race_position']}")
        
        for feature in momentum_features:
            value = row[feature]
            print(f"   {feature}: {value:.2f}" if not pd.isna(value) else f"   {feature}: NaN")
    
    # Verificar estrategias de NaN handling
    print(f"\n🔍 VERIFICACIÓN DE NaN HANDLING:")
    print("=" * 40)
    
    for feature in momentum_features:
        if feature in rookie_data.columns:
            nan_count = rookie_data[feature].isna().sum()
            total_count = len(rookie_data)
            print(f"   {feature}: {nan_count}/{total_count} NaN ({(nan_count/total_count)*100:.1f}%)")
    
    # Mostrar valores de fallback
    print(f"\n📊 ESTRATEGIAS DE FALLBACK:")
    print("=" * 30)
    
    # Promedio general del piloto rookie
    rookie_avg_position = rookie_data['race_position'].mean()
    print(f"   Promedio general Bearman: {rookie_avg_position:.2f}")
    
    # Promedio del equipo
    team_avg = df_enhanced[df_enhanced['team'] == 'Ferrari']['race_position'].mean()
    print(f"   Promedio equipo Ferrari: {team_avg:.2f}")
    
    # Promedio general del dataset
    overall_avg = df_enhanced['race_position'].mean()
    print(f"   Promedio general dataset: {overall_avg:.2f}")

if __name__ == "__main__":
    test_rookie_features()
