"""
Test completo del manejo de rookies con todas las estrategias de fallback
"""

import pandas as pd
import numpy as np
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

def test_complete_rookie_handling():
    """Prueba completa del manejo de rookies incluyendo fallbacks"""
    
    print("🧪 TEST COMPLETO DE MANEJO DE ROOKIES")
    print("=" * 55)
    
    # Crear datos más realistas
    data = []
    
    # Veteranos con historial completo
    for race_num in range(1, 8):  # 7 carreras
        for driver_info in [
            ('Hamilton', 'Mercedes', [2, 3, 1, 4, 3, 2, 5]),
            ('Verstappen', 'Red Bull Racing', [1, 1, 2, 1, 1, 1, 1]),
            ('Leclerc', 'Ferrari', [3, 4, 5, 3, 4, 3, 3])
        ]:
            driver, team, positions = driver_info
            data.append({
                'driver': driver,
                'race_name': f'Bahrain_GP' if race_num == 1 else f'Race_{race_num}',
                'year': 2024,
                'race_position': positions[race_num-1],
                'quali_position': positions[race_num-1] + np.random.randint(-2, 3),
                'points': max(0, 26 - positions[race_num-1]*2),
                'team': team
            })
    
    # Rookie 1: Aparece solo en las últimas 3 carreras
    rookie_races = [5, 6, 7]
    for race_num in rookie_races:
        data.append({
            'driver': 'Bearman',  # Rookie total
            'race_name': f'Race_{race_num}',
            'year': 2024,
            'race_position': 16 + (race_num - 5),  # Mejorando: 16, 17, 18
            'quali_position': 18 + (race_num - 5),
            'points': 0,
            'team': 'Ferrari'
        })
    
    # Rookie 2: Aparece desde carrera 3 (rookie de mitad de temporada)
    for race_num in range(3, 8):
        data.append({
            'driver': 'Sargeant',  # Rookie con algo más de experiencia
            'race_name': f'Race_{race_num}',
            'year': 2024,
            'race_position': 19 - (race_num - 3),  # Mejorando: 19, 18, 17, 16, 15
            'quali_position': 20 - (race_num - 3),
            'points': 1 if race_num >= 6 else 0,  # Suma algunos puntos al final
            'team': 'Williams'
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['driver', 'race_name']).reset_index(drop=True)
    
    print(f"📊 DATOS INICIALES:")
    for driver, count in df['driver'].value_counts().items():
        driver_data = df[df['driver'] == driver]
        first_race = driver_data['race_name'].iloc[0]
        last_race = driver_data['race_name'].iloc[-1]
        print(f"   {driver}: {count} carreras ({first_race} → {last_race})")
    
    # Aplicar feature engineering completo
    engineer = AdvancedFeatureEngineer()
    print(f"\n🔧 Aplicando feature engineering completo...")
    
    df_enhanced = engineer.create_all_advanced_features(df.copy())
    
    # Analizar cada rookie
    rookies = ['Bearman', 'Sargeant']
    
    for rookie in rookies:
        print(f"\n👨‍🚀 ANÁLISIS ROOKIE: {rookie}")
        print("=" * 45)
        
        rookie_data = df_enhanced[df_enhanced['driver'] == rookie].copy()
        
        print(f"   📅 Carreras: {len(rookie_data)}")
        print(f"   🏁 Posiciones: {list(rookie_data['race_position'])}")
        
        # Features de momentum más importantes
        key_features = [
            'avg_position_last_3',
            'points_last_3', 
            'position_trend_last_5'
        ]
        
        print(f"\n   📈 FEATURES DE MOMENTUM:")
        for feature in key_features:
            if feature in rookie_data.columns:
                values = rookie_data[feature].values
                nan_count = pd.isna(values).sum()
                valid_values = values[~pd.isna(values)]
                
                print(f"      {feature}:")
                print(f"         Valores: {[f'{v:.2f}' if not pd.isna(v) else 'NaN' for v in values]}")
                print(f"         NaN: {nan_count}/{len(values)} ({(nan_count/len(values))*100:.1f}%)")
                if len(valid_values) > 0:
                    print(f"         Promedio válido: {valid_values.mean():.2f}")
        
        # Features históricas en circuitos
        historical_features = [f for f in engineer.created_features if 'track_avg' in f]
        
        print(f"\n   🏁 FEATURES HISTÓRICAS EN CIRCUITOS:")
        for feature in historical_features[:3]:  # Solo las primeras 3
            if feature in rookie_data.columns:
                values = rookie_data[feature].values
                print(f"      {feature}: {[f'{v:.2f}' if not pd.isna(v) else 'NaN' for v in values]}")
    
    # Mostrar estrategias de fallback aplicadas
    print(f"\n🔄 VERIFICACIÓN DE FALLBACKS APLICADOS:")
    print("=" * 45)
    
    # Después del procesamiento completo, verificar NaN restantes
    for rookie in rookies:
        rookie_data = df_enhanced[df_enhanced['driver'] == rookie]
        rookie_features = [f for f in engineer.created_features if f in rookie_data.columns]
        
        total_values = len(rookie_data) * len(rookie_features)
        nan_values = rookie_data[rookie_features].isnull().sum().sum()
        
        print(f"   {rookie}:")
        print(f"      Total valores: {total_values}")
        print(f"      NaN restantes: {nan_values}")
        print(f"      Completitud: {((total_values - nan_values) / total_values) * 100:.1f}%")
        
        # Mostrar algunas features con sus valores finales
        sample_features = ['avg_position_last_3', 'points_last_3', 'driver_track_avg_position']
        for feature in sample_features:
            if feature in rookie_data.columns:
                final_values = rookie_data[feature].values
                print(f"      {feature}: {[f'{v:.2f}' for v in final_values]}")

if __name__ == "__main__":
    test_complete_rookie_handling()
