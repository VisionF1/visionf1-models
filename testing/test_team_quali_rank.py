"""
Test específico para verificar team_quali_rank corregido
"""

import pandas as pd
import numpy as np
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

def test_team_quali_rank():
    print("🔧 TESTING TEAM_QUALI_RANK CORREGIDO")
    print("=" * 50)
    
    # Cargar datos reales
    df = pd.read_pickle('app/models_cache/cached_data.pkl')
    print(f"📊 Dataset cargado: {df.shape}")
    
    # Crear feature engineer
    fe = AdvancedFeatureEngineer()
    
    # Aplicar features
    enhanced_df = fe.create_all_advanced_features(df.copy())
    
    # Analizar team_quali_rank
    if 'team_quali_rank' in enhanced_df.columns:
        print(f"\n📊 ANÁLISIS DE TEAM_QUALI_RANK:")
        print("=" * 50)
        
        # Verificar rango de valores
        rank_data = enhanced_df['team_quali_rank'].dropna()
        print(f"Rango de valores: {rank_data.min()} - {rank_data.max()}")
        print(f"Valores únicos: {sorted(rank_data.unique())}")
        
        # Ejemplo de una carrera específica
        sample_race = enhanced_df.groupby(['race_name', 'year']).first().index[0]
        race_name, year = sample_race
        
        race_data = enhanced_df[
            (enhanced_df['race_name'] == race_name) & 
            (enhanced_df['year'] == year)
        ][['team', 'driver', 'quali_position', 'team_quali_rank']].sort_values('team_quali_rank')
        
        print(f"\n🏁 EJEMPLO: {race_name} {year}")
        print("=" * 50)
        print("Team Quali Rank | Team | Driver | Quali Pos")
        print("-" * 50)
        
        for _, row in race_data.head(10).iterrows():
            print(f"{row['team_quali_rank']:13.0f} | {row['team'][:15]:15} | {row['driver'][:15]:15} | {row['quali_position']:9.0f}")
        
        # Ranking promedio por equipo en todas las carreras
        print(f"\n📈 RANKING PROMEDIO POR EQUIPO (todas las carreras):")
        print("=" * 50)
        
        team_avg_rank = enhanced_df.groupby('team')['team_quali_rank'].agg(['mean', 'count']).sort_values('mean')
        
        print("Equipo | Ranking Promedio | Carreras")
        print("-" * 50)
        
        for team, stats in team_avg_rank.iterrows():
            print(f"{team[:20]:20} | {stats['mean']:15.2f} | {stats['count']:8.0f}")
        
        # Verificar correlación con posiciones reales
        print(f"\n🔍 VALIDACIÓN:")
        print("=" * 50)
        
        # Para cada carrera, verificar que el equipo con mejor team_quali_rank 
        # tenga efectivamente la mejor posición de clasificación
        validation_results = []
        
        for (race, year), group in enhanced_df.groupby(['race_name', 'year']):
            if len(group) > 5:  # Solo carreras con suficientes datos
                # Mejor team_quali_rank de la carrera
                best_rank_team = group.loc[group['team_quali_rank'].idxmin(), 'team']
                
                # Mejor posición real de clasificación de la carrera
                best_quali_team = group.loc[group['quali_position'].idxmin(), 'team']
                
                validation_results.append({
                    'race': f"{race} {year}",
                    'rank_match': best_rank_team == best_quali_team,
                    'best_rank_team': best_rank_team,
                    'best_quali_team': best_quali_team
                })
        
        match_rate = sum(1 for r in validation_results if r['rank_match']) / len(validation_results)
        print(f"✅ Precisión: {match_rate:.1%} de carreras donde el mejor team_quali_rank")
        print(f"   coincide con el equipo que tuvo la pole position")
        
        # Mostrar algunos casos de no coincidencia para análisis
        mismatches = [r for r in validation_results if not r['rank_match']]
        if mismatches:
            print(f"\n⚠️  Algunos casos de no coincidencia (pueden ser válidos):")
            for i, case in enumerate(mismatches[:3]):
                print(f"   {case['race']}: Mejor rank={case['best_rank_team']}, Pole={case['best_quali_team']}")
    
    else:
        print("❌ team_quali_rank no encontrada!")

if __name__ == "__main__":
    test_team_quali_rank()
