"""
Enhanced debut handling - Versión mejorada para manejar debuts absolutos
"""

import pandas as pd
import numpy as np

def create_debut_aware_momentum_features(df):
    """
    Crea features de momentum que manejan inteligentemente los debuts absolutos
    """
    
    print("🔧 Creando features de momentum con manejo de debuts...")
    
    df = df.copy()
    df = df.sort_values(['driver', 'year', 'race_name'])
    
    created_features = []
    
    # ESTRATEGIA MEJORADA para debuts absolutos
    if 'race_position' in df.columns:
        
        # 1. Features básicas con shift (como antes)
        df['avg_position_last_3'] = df.groupby('driver')['race_position'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        
        # 2. NUEVA: Features para debuts (sin shift, usando expectativas)
        df['debut_expectation'] = df.groupby('team')['race_position'].transform('median')
        
        # 3. Combinar: Si es debut (NaN en avg_position_last_3), usar expectativa del equipo
        df['avg_position_last_3_enhanced'] = df['avg_position_last_3'].fillna(df['debut_expectation'])
        
        # 4. Marcar debuts para features especiales
        df['is_debut'] = df['avg_position_last_3'].isna().astype(int)
        
        created_features.extend(['avg_position_last_3_enhanced', 'debut_expectation', 'is_debut'])
    
    if 'points' in df.columns:
        # Para puntos en debut, usar 0 (realista para debuts)
        df['points_last_3'] = df.groupby('driver')['points'].transform(
            lambda x: x.rolling(3, min_periods=1).sum().shift(1)
        )
        df['points_last_3_enhanced'] = df['points_last_3'].fillna(0)  # Debuts raramente puntúan
        created_features.append('points_last_3_enhanced')
    
    print(f"   ✅ {len(created_features)} features de momentum mejoradas creadas")
    
    return df, created_features

def test_debut_aware_features():
    """Prueba las features mejoradas para debuts"""
    
    print("🧪 TEST DE FEATURES MEJORADAS PARA DEBUTS")
    print("=" * 50)
    
    # Datos de prueba
    data = []
    
    # Veterano con historial
    for race_num in range(1, 4):
        data.append({
            'driver': 'Hamilton',
            'team': 'Mercedes',
            'year': 2024,
            'race_name': f'Race_{race_num}',
            'race_position': [2, 3, 1][race_num-1],
            'points': [18, 15, 25][race_num-1]
        })
    
    # Debut absoluto
    data.append({
        'driver': 'Colapinto',  # DEBUT ABSOLUTO
        'team': 'Williams',
        'year': 2024,
        'race_name': 'Race_1',
        'race_position': 18,
        'points': 0
    })
    
    # Equipo de referencia para expectativas
    for race_num in range(1, 3):
        data.append({
            'driver': 'Sargeant',
            'team': 'Williams',  # Mismo equipo que el debutante
            'year': 2024,
            'race_name': f'Race_{race_num}',
            'race_position': [19, 17][race_num-1],
            'points': [0, 0][race_num-1]
        })
    
    df = pd.DataFrame(data)
    
    print("📊 DATOS DE PRUEBA:")
    for driver in df['driver'].unique():
        driver_data = df[df['driver'] == driver]
        print(f"   {driver}: {len(driver_data)} carreras")
    
    # Calcular expectativas de equipo
    team_expectations = df.groupby('team')['race_position'].median()
    print(f"\n📈 EXPECTATIVAS POR EQUIPO:")
    for team, expectation in team_expectations.items():
        print(f"   {team}: posición {expectation:.1f}")
    
    # Aplicar features mejoradas
    df_enhanced, created_features = create_debut_aware_momentum_features(df)
    
    # Analizar resultados
    print(f"\n👶 ANÁLISIS DEL DEBUTANTE:")
    debut_data = df_enhanced[df_enhanced['driver'] == 'Colapinto']
    
    if len(debut_data) > 0:
        debut_row = debut_data.iloc[0]
        print(f"   Piloto: {debut_row['driver']}")
        print(f"   Equipo: {debut_row['team']}")
        print(f"   Posición real: {debut_row['race_position']}")
        print(f"   Es debut: {'✅ Sí' if debut_row['is_debut'] else '❌ No'}")
        print(f"   Expectativa equipo: {debut_row['debut_expectation']:.1f}")
        print(f"   avg_position_last_3 original: {debut_row['avg_position_last_3']}")
        print(f"   avg_position_last_3 mejorada: {debut_row['avg_position_last_3_enhanced']:.1f}")
        print(f"   points_last_3 original: {debut_row['points_last_3']}")
        print(f"   points_last_3 mejorada: {debut_row['points_last_3_enhanced']:.1f}")
    
    # Comparar con veterano
    print(f"\n🏆 ANÁLISIS DEL VETERANO (Hamilton):")
    veteran_data = df_enhanced[df_enhanced['driver'] == 'Hamilton']
    
    for idx, row in veteran_data.iterrows():
        race = row['race_name']
        is_debut = row['is_debut']
        enhanced_avg = row['avg_position_last_3_enhanced']
        print(f"   {race}: debut={is_debut}, avg_enhanced={enhanced_avg:.1f}")

if __name__ == "__main__":
    test_debut_aware_features()
