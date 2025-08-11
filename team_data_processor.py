#!/usr/bin/env python3
"""
Data Processor with Team Mapping - Procesador de datos con mapeo histórico de equipos
Aplica transformaciones históricas para mantener consistencia temporal
"""

import pandas as pd
import pickle
import os
from datetime import datetime
import sys

# Agregar el path para importar el team_mapper
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.core.utils.team_mapper import F1TeamMapper


class F1DataProcessor:
    """
    Procesador de datos F1 que aplica mapeo histórico de equipos
    para mantener consistencia en análisis longitudinales.
    """
    
    def __init__(self):
        self.team_mapper = F1TeamMapper()
        self.cache_dir = "app/models_cache"
        self.raw_data_dir = os.path.join(self.cache_dir, "raw_data")
        self.processed_data_dir = os.path.join(self.cache_dir, "processed_data")
        
        # Crear directorio procesado si no existe
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_all_raw_data(self):
        """Carga todos los datos raw sin procesar"""
        print("📦 CARGANDO DATOS RAW...")
        
        all_data = []
        files_loaded = 0
        
        if not os.path.exists(self.raw_data_dir):
            print(f"❌ No se encuentra el directorio: {self.raw_data_dir}")
            return None
            
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith('.pkl') and 'complete' in filename:
                file_path = os.path.join(self.raw_data_dir, filename)
                try:
                    with open(file_path, 'rb') as f:
                        race_data = pickle.load(f)
                        if isinstance(race_data, dict):
                            if 'data' in race_data:
                                race_df = race_data['data']
                            elif 'race_data' in race_data:
                                race_df = race_data['race_data']
                            else:
                                race_df = None
                                
                            if race_df is not None and not race_df.empty:
                                all_data.append(race_df)
                                files_loaded += 1
                        elif isinstance(race_data, pd.DataFrame):
                            if not race_data.empty:
                                all_data.append(race_data)
                                files_loaded += 1
                except Exception as e:
                    print(f"   ⚠️  Error cargando {filename}: {e}")
        
        if all_data:
            print(f"✅ {files_loaded} archivos cargados")
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            print("❌ No se encontraron datos válidos")
            return None
    
    def apply_team_mapping(self, df):
        """
        Aplica el mapeo histórico de equipos al DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos originales
            
        Returns:
            pandas.DataFrame: DataFrame con equipos mapeados
        """
        print("🔄 APLICANDO MAPEO HISTÓRICO DE EQUIPOS...")
        
        if 'team' not in df.columns:
            print("⚠️  No se encontró columna 'team'")
            return df
        
        # Crear copia y preservar nombres originales
        df_processed = df.copy()
        
        # Guardar nombres originales para trazabilidad
        df_processed['team_original'] = df_processed['team'].copy()
        
        # Aplicar mapeo
        df_processed['team'] = df_processed['team'].map(self.team_mapper.map_to_current)
        
        # Agregar información de linaje para análisis
        df_processed['team_lineage'] = df_processed['team_original'].apply(
            lambda x: self.team_mapper.map_to_current(x)
        )
        
        # Estadísticas del mapeo
        original_teams = df_processed['team_original'].nunique()
        mapped_teams = df_processed['team'].nunique()
        
        print(f"   📊 Equipos originales: {original_teams}")
        print(f"   📊 Equipos después del mapeo: {mapped_teams}")
        print(f"   📊 Reducción: {original_teams - mapped_teams} equipos consolidados")
        
        # Mostrar transformaciones aplicadas
        transformations = df_processed[df_processed['team_original'] != df_processed['team']]
        if not transformations.empty:
            print("   🔄 Transformaciones aplicadas:")
            for original, mapped in transformations[['team_original', 'team']].drop_duplicates().values:
                count = len(transformations[transformations['team_original'] == original])
                print(f"      {original} -> {mapped} ({count} registros)")
        
        return df_processed
    
    def generate_team_consistency_report(self, df):
        """
        Genera reporte de consistencia de equipos a través del tiempo.
        
        Args:
            df (pandas.DataFrame): DataFrame procesado con mapeo
            
        Returns:
            dict: Reporte de consistencia
        """
        print("\n📋 GENERANDO REPORTE DE CONSISTENCIA TEMPORAL...")
        
        if not all(col in df.columns for col in ['team', 'team_original', 'year']):
            print("⚠️  Faltan columnas necesarias para el reporte")
            return {}
        
        # Análisis por año
        yearly_analysis = {}
        years = sorted(df['year'].unique())
        
        for year in years:
            year_data = df[df['year'] == year]
            yearly_analysis[year] = {
                'teams_original': sorted(year_data['team_original'].unique()),
                'teams_mapped': sorted(year_data['team'].unique()),
                'records': len(year_data)
            }
        
        # Análisis de linajes
        lineage_analysis = {}
        for current_team in self.team_mapper.get_all_teams_2025():
            historical_names = self.team_mapper.get_historical_names(current_team)
            
            # Registros por cada nombre histórico
            lineage_records = {}
            for historical_name in historical_names:
                records = df[df['team_original'] == historical_name]
                if not records.empty:
                    lineage_records[historical_name] = {
                        'records': len(records),
                        'years': sorted(records['year'].unique()),
                        'races': records['race_name'].nunique() if 'race_name' in records.columns else 0
                    }
            
            if lineage_records:
                lineage_analysis[current_team] = lineage_records
        
        report = {
            'yearly_analysis': yearly_analysis,
            'lineage_analysis': lineage_analysis,
            'total_records': len(df),
            'years_covered': years,
            'consolidation_summary': self.team_mapper.get_transformation_summary()
        }
        
        return report
    
    def save_processed_data(self, df, consistency_report):
        """
        Guarda los datos procesados y el reporte de consistencia.
        
        Args:
            df (pandas.DataFrame): DataFrame procesado
            consistency_report (dict): Reporte de consistencia
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar DataFrame procesado
        processed_file = os.path.join(self.processed_data_dir, f"f1_data_team_mapped_{timestamp}.pkl")
        with open(processed_file, 'wb') as f:
            pickle.dump(df, f)
        
        # Guardar como CSV también para fácil inspección
        csv_file = os.path.join(self.processed_data_dir, f"f1_data_team_mapped_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        # Guardar reporte de consistencia
        report_file = os.path.join(self.processed_data_dir, f"team_consistency_report_{timestamp}.pkl")
        with open(report_file, 'wb') as f:
            pickle.dump(consistency_report, f)
        
        print(f"\n💾 DATOS PROCESADOS GUARDADOS:")
        print(f"   📊 DataFrame: {processed_file}")
        print(f"   📄 CSV: {csv_file}")
        print(f"   📋 Reporte: {report_file}")
        
        return processed_file, csv_file, report_file
    
    def print_consistency_report(self, report):
        """Imprime un resumen del reporte de consistencia."""
        print("\n📊 REPORTE DE CONSISTENCIA TEMPORAL")
        print("=" * 60)
        
        print(f"📈 Años cubiertos: {min(report['years_covered'])} - {max(report['years_covered'])}")
        print(f"📦 Total registros: {report['total_records']:,}")
        
        print("\n🏁 ANÁLISIS POR LINAJE DE EQUIPOS:")
        for current_team, lineage_data in report['lineage_analysis'].items():
            print(f"\n🏎️  {current_team}:")
            total_records = sum(data['records'] for data in lineage_data.values())
            print(f"   📊 Total registros: {total_records:,}")
            
            for historical_name, data in lineage_data.items():
                years_str = f"{min(data['years'])}-{max(data['years'])}" if len(data['years']) > 1 else str(data['years'][0])
                symbol = "✅" if historical_name == current_team else "📅"
                print(f"   {symbol} {historical_name}: {data['records']:,} registros ({years_str})")
        
        print(f"\n🔄 CONSOLIDACIÓN APLICADA:")
        summary = report['consolidation_summary']
        print(f"   📊 Equipos con transformaciones: {summary['teams_with_changes']}")
        print(f"   ✅ Equipos estables: {summary['stable_teams_count']}")
        print(f"   🎯 Total equipos finales: {summary['total_teams_2025']}")
    
    def process_all_data(self):
        """Procesa todos los datos aplicando mapeo histórico completo."""
        print("🏁 F1 DATA PROCESSOR CON MAPEO HISTÓRICO")
        print("🔄 Iniciando procesamiento completo...")
        print("=" * 60)
        
        # 1. Cargar datos raw
        raw_data = self.load_all_raw_data()
        if raw_data is None:
            return None
        
        print(f"📊 Datos cargados: {len(raw_data):,} registros, {len(raw_data.columns)} columnas")
        
        # 2. Mostrar resumen del mapper
        print("\n" + "="*60)
        self.team_mapper.print_mapping_summary()
        
        # 3. Aplicar mapeo
        print("\n" + "="*60)
        processed_data = self.apply_team_mapping(raw_data)
        
        # 4. Generar reporte de consistencia
        consistency_report = self.generate_team_consistency_report(processed_data)
        
        # 5. Mostrar reporte
        self.print_consistency_report(consistency_report)
        
        # 6. Guardar resultados
        files = self.save_processed_data(processed_data, consistency_report)
        
        print(f"\n🎉 ¡PROCESAMIENTO COMPLETADO!")
        print(f"📊 Dataset procesado: {len(processed_data):,} registros")
        print(f"🔄 Equipos consolidados de {consistency_report['consolidation_summary']['teams_with_changes']} transformaciones")
        
        return processed_data, consistency_report, files


def main():
    """Función principal para ejecutar el procesamiento completo."""
    processor = F1DataProcessor()
    result = processor.process_all_data()
    
    if result:
        processed_data, consistency_report, files = result
        print(f"\n✅ Procesamiento exitoso. Archivos generados:")
        for file in files:
            print(f"   📁 {file}")


if __name__ == "__main__":
    main()
