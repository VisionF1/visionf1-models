"""
Test de Features Avanzadas con Datos Reales
Prueba las 12 features nuevas implementadas y muestra ejemplos de valores
"""

import pandas as pd
import numpy as np
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer

def test_new_features():
    print("🏎️ TESTING FEATURES AVANZADAS CON DATOS REALES")
    print("=" * 60)
    
    # Cargar datos reales
    try:
        print("📂 Cargando dataset...")
        # Intentar cargar desde diferentes fuentes disponibles
        dataset_paths = [
            'app/models_cache/weather_enhanced_dataset.csv',
            'app/models_cache/cached_data.pkl'
        ]
        
        df = None
        for path in dataset_paths:
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                elif path.endswith('.pkl'):
                    df = pd.read_pickle(path)
                print(f"   ✅ Dataset cargado desde {path}: {df.shape}")
                break
            except:
                continue
        
        if df is None:
            print("   ❌ No se encontraron datasets disponibles")
            return
        
        # Crear instancia del feature engineer
        fe = AdvancedFeatureEngineer()
        print("🔧 Aplicando feature engineering avanzado...")
        
        # Aplicar todas las features
        enhanced_df = fe.create_all_advanced_features(df.copy())
        print(f"   ✅ Features aplicadas: {enhanced_df.shape}")
        
        # Features solicitadas específicamente
        requested_features = [
            'quali_gap_to_pole',
            'fp1_gap_to_fastest', 
            'team_quali_rank',
            'avg_position_last_3',
            'points_last_3',
            'heat_index',
            'weather_difficulty_index',
            'team_track_avg_position',
            'fp1_to_quali_improvement',
            'sector_consistency',
            'grid_to_race_change',
            'overtaking_ability'
        ]
        
        print(f"\n🎯 ANÁLISIS DE LAS 12 FEATURES SOLICITADAS:")
        print("=" * 60)
        
        for i, feature in enumerate(requested_features, 1):
            if feature in enhanced_df.columns:
                data = enhanced_df[feature].dropna()
                
                if len(data) > 0:
                    print(f"\n{i:2d}. 📊 {feature}")
                    print(f"    Registros válidos: {len(data):,}/{len(enhanced_df):,} ({len(data)/len(enhanced_df)*100:.1f}%)")
                    print(f"    Rango: {data.min():.3f} - {data.max():.3f}")
                    print(f"    Promedio: {data.mean():.3f}")
                    print(f"    Mediana: {data.median():.3f}")
                    
                    # Mostrar algunos ejemplos reales
                    print(f"    Ejemplos: {', '.join([f'{x:.3f}' for x in data.head(3).values])}")
                else:
                    print(f"\n{i:2d}. ❌ {feature}: Sin datos válidos")
            else:
                print(f"\n{i:2d}. ❌ {feature}: NO ENCONTRADA")
        
        # Análisis específico de weather_difficulty_index
        print(f"\n🌤️ ANÁLISIS DETALLADO: weather_difficulty_index")
        print("=" * 60)
        
        if 'weather_difficulty_index' in enhanced_df.columns:
            weather_data = enhanced_df[['weather_difficulty_index', 'session_rainfall', 
                                      'session_humidity', 'session_air_temp']].dropna()
            
            if len(weather_data) > 0:
                print(f"Registros con datos meteorológicos completos: {len(weather_data)}")
                
                # Encontrar casos extremos
                high_difficulty = weather_data[weather_data['weather_difficulty_index'] > 3]
                low_difficulty = weather_data[weather_data['weather_difficulty_index'] < 1]
                
                print(f"\n🔴 Condiciones DIFÍCILES (índice > 3): {len(high_difficulty)} casos")
                if len(high_difficulty) > 0:
                    example = high_difficulty.iloc[0]
                    print(f"   Ejemplo - Índice: {example['weather_difficulty_index']:.2f}")
                    print(f"            Lluvia: {example['session_rainfall']}")
                    print(f"            Humedad: {example['session_humidity']:.1f}%")
                    print(f"            Temperatura: {example['session_air_temp']:.1f}°C")
                
                print(f"\n🟢 Condiciones IDEALES (índice < 1): {len(low_difficulty)} casos")
                if len(low_difficulty) > 0:
                    example = low_difficulty.iloc[0]
                    print(f"   Ejemplo - Índice: {example['weather_difficulty_index']:.2f}")
                    print(f"            Lluvia: {example['session_rainfall']}")
                    print(f"            Humedad: {example['session_humidity']:.1f}%")
                    print(f"            Temperatura: {example['session_air_temp']:.1f}°C")
        
        # Análisis de heat_index
        print(f"\n🌡️ ANÁLISIS DETALLADO: heat_index")
        print("=" * 60)
        
        if 'heat_index' in enhanced_df.columns:
            heat_data = enhanced_df[['heat_index', 'session_air_temp', 'session_humidity']].dropna()
            
            if len(heat_data) > 0:
                extreme_heat = heat_data[heat_data['heat_index'] > 35]
                cool_conditions = heat_data[heat_data['heat_index'] < 25]
                
                print(f"🔥 Condiciones CALUROSAS (índice > 35): {len(extreme_heat)} casos")
                if len(extreme_heat) > 0:
                    example = extreme_heat.iloc[0]
                    print(f"   Ejemplo - Heat Index: {example['heat_index']:.1f}")
                    print(f"            Temperatura: {example['session_air_temp']:.1f}°C")
                    print(f"            Humedad: {example['session_humidity']:.1f}%")
                
                print(f"❄️ Condiciones FRESCAS (índice < 25): {len(cool_conditions)} casos")
                if len(cool_conditions) > 0:
                    example = cool_conditions.iloc[0]
                    print(f"   Ejemplo - Heat Index: {example['heat_index']:.1f}")
                    print(f"            Temperatura: {example['session_air_temp']:.1f}°C")
                    print(f"            Humedad: {example['session_humidity']:.1f}%")
        
        # Análisis de team_quali_rank
        print(f"\n🏁 ANÁLISIS DETALLADO: team_quali_rank")
        print("=" * 60)
        
        if 'team_quali_rank' in enhanced_df.columns:
            rank_data = enhanced_df[['team_quali_rank', 'team']].dropna()
            
            if len(rank_data) > 0:
                # Ranking promedio por equipo
                team_avg_rank = rank_data.groupby('team')['team_quali_rank'].mean().sort_values()
                
                print("📊 Ranking promedio de equipos en clasificación:")
                for i, (team, avg_rank) in enumerate(team_avg_rank.head(10).items(), 1):
                    print(f"   {i:2d}. {team}: {avg_rank:.2f}")
        
        # Test de integración con EnhancedDataPreparer
        print(f"\n🔧 TEST DE INTEGRACIÓN CON ENHANCED DATA PREPARER")
        print("=" * 60)
        
        preparer = EnhancedDataPreparer(use_advanced_features=True)
        X, y, encoder, feature_names = preparer.prepare_enhanced_features(df.copy())
        
        if X is not None:
            print(f"\n✅ INTEGRACIÓN EXITOSA:")
            print(f"   📊 Shape final: {X.shape}")
            print(f"   🎯 Features seleccionadas: {len(feature_names)}")
            
            # Verificar cuáles de nuestras features están incluidas
            included_new_features = [f for f in requested_features if f in feature_names]
            print(f"   🆕 Features nuevas incluidas: {len(included_new_features)}")
            
            for feature in included_new_features:
                print(f"      ✅ {feature}")
        
        print(f"\n🎉 TEST COMPLETADO EXITOSAMENTE!")
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_features()
