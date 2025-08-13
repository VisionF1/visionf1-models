"""
Test final del sistema completo con debuts absolutos
"""

from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
import pandas as pd
import numpy as np

def test_complete_system_with_debuts():
    """Prueba el sistema completo con debuts absolutos"""
    
    print("🧪 TEST SISTEMA COMPLETO CON DEBUTS ABSOLUTOS")
    print("=" * 55)
    
    # Crear datos más completos para el test
    data = []
    
    # Veteranos para establecer baseline
    for race_num in range(1, 4):
        for driver_info in [
            ('Hamilton', 'Mercedes', [2, 3, 1], [2.1, 3.2, 1.8]),
            ('Verstappen', 'Red Bull Racing', [1, 1, 2], [1.9, 1.7, 2.1]),
            ('Leclerc', 'Ferrari', [3, 4, 5], [3.1, 4.2, 5.3]),
        ]:
            driver, team, positions, quali_times = driver_info
            data.append({
                'driver': driver,
                'race_name': f'Race_{race_num}',
                'year': 2024,
                'race_position': positions[race_num-1],
                'quali_position': positions[race_num-1],
                'quali_best_time': quali_times[race_num-1],
                'race_best_lap_time': quali_times[race_num-1] + 1.5,
                'clean_air_pace': quali_times[race_num-1] + 0.8,
                'grid_position': positions[race_num-1],
                'points': max(0, 26 - positions[race_num-1]*2),
                'team': team,
                'session_air_temp': 25.0,
                'session_track_temp': 35.0,
                'session_humidity': 60.0,
                'session_rainfall': 0
            })
    
    # DEBUT ABSOLUTO
    data.append({
        'driver': 'Colapinto',  # DEBUT ABSOLUTO
        'race_name': 'Race_1',
        'year': 2024,
        'race_position': 18,
        'quali_position': 19,
        'quali_best_time': 5.2,  # Tiempo más lento
        'race_best_lap_time': 6.8,
        'clean_air_pace': 6.0,
        'grid_position': 19,
        'points': 0,
        'team': 'Williams',
        'session_air_temp': 25.0,
        'session_track_temp': 35.0,
        'session_humidity': 60.0,
        'session_rainfall': 0
    })
    
    df = pd.DataFrame(data)
    
    print(f"📊 DATOS DE PRUEBA:")
    print(f"   Total registros: {len(df)}")
    print(f"   Pilotos únicos: {df['driver'].nunique()}")
    debuts = df[df['driver'] == 'Colapinto']
    print(f"   Debut absoluto: {debuts['driver'].iloc[0]} en {debuts['race_name'].iloc[0]}")
    
    # Usar el enhanced data preparer
    preparer = EnhancedDataPreparer(use_advanced_features=True)
    
    print(f"\n🚀 Procesando con Enhanced Data Preparer...")
    try:
        X, y, label_encoder, feature_names = preparer.prepare_enhanced_features(df)
        
        if X is not None:
            print(f"\n✅ PROCESAMIENTO EXITOSO")
            print(f"   📊 Shape final: {X.shape}")
            print(f"   🎯 Features totales: {len(feature_names)}")
            
            # Analizar el debutante en los datos finales
            debut_index = df[df['driver'] == 'Colapinto'].index[0]
            
            if debut_index < len(X):
                debut_features = X.iloc[debut_index]
                
                print(f"\n👶 FEATURES DEL DEBUTANTE EN DATOS FINALES:")
                print("=" * 45)
                
                # Mostrar features clave
                key_features = [
                    'avg_position_last_3', 'points_last_3', 'is_debut',
                    'driver_track_avg_position', 'team_encoded'
                ]
                
                for feature in key_features:
                    if feature in feature_names:
                        idx = feature_names.index(feature)
                        value = debut_features.iloc[idx]
                        print(f"   {feature}: {value}")
                
                # Verificar que no hay NaN
                nan_count = debut_features.isna().sum()
                print(f"\n   📊 Completitud debut: {((len(debut_features) - nan_count) / len(debut_features) * 100):.1f}%")
                print(f"   🎯 NaN restantes: {nan_count}")
                
                if nan_count == 0:
                    print(f"   ✅ DEBUT PERFECTAMENTE MANEJADO - SIN NaN")
                else:
                    print(f"   ⚠️  Aún hay {nan_count} NaN en el debut")
            
            # Mostrar resumen de features por categoría
            print(preparer.get_feature_importance_summary())
            
        else:
            print(f"❌ Error en el procesamiento")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_system_with_debuts()
