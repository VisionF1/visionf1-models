#!/usr/bin/env python3
"""
Explorador de Datos F1 - Análisis completo del dataset
Analiza todas las características disponibles, distribuciones y estadísticas
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class F1DataExplorer:
    def __init__(self):
        self.cache_dir = "app/models_cache"
        self.raw_data_dir = os.path.join(self.cache_dir, "raw_data")
        self.data = None
        self.analysis_results = {}
        
    def load_all_data(self):
        """Carga todos los datos disponibles"""
        print("🔍 CARGANDO TODOS LOS DATOS DISPONIBLES")
        print("=" * 60)
        
        all_data = []
        files_loaded = 0
        
        if not os.path.exists(self.raw_data_dir):
            print(f"❌ No se encuentra el directorio: {self.raw_data_dir}")
            return None
            
        # Buscar todos los archivos .pkl en raw_data
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith('.pkl') and 'complete' in filename:
                file_path = os.path.join(self.raw_data_dir, filename)
                try:
                    print(f"📦 Cargando: {filename}")
                    with open(file_path, 'rb') as f:
                        race_data = pickle.load(f)
                        if isinstance(race_data, dict):
                            # Intentar diferentes claves posibles
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
            print(f"\n✅ {files_loaded} archivos cargados exitosamente")
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"📊 Dataset final: {len(self.data)} filas, {len(self.data.columns)} columnas")
            return self.data
        else:
            print("❌ No se encontraron datos válidos")
            return None
    
    def general_overview(self):
        """Análisis general del dataset"""
        if self.data is None:
            print("❌ No hay datos cargados")
            return
            
        print("\n🔍 RESUMEN GENERAL DEL DATASET")
        print("=" * 60)
        
        # Información básica
        print(f"📊 Dimensiones: {self.data.shape[0]:,} filas × {self.data.shape[1]} columnas")
        print(f"💾 Tamaño en memoria: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Información temporal
        if 'year' in self.data.columns:
            years = sorted(self.data['year'].unique())
            print(f"📅 Años disponibles: {years[0]} - {years[-1]} ({len(years)} años)")
            
            # Distribución por año
            year_counts = self.data['year'].value_counts().sort_index()
            print(f"\n📈 Distribución por año:")
            for year, count in year_counts.items():
                percentage = (count / len(self.data)) * 100
                print(f"   {year}: {count:,} registros ({percentage:.1f}%)")
        
        # Información de carreras
        if 'race_name' in self.data.columns:
            races = self.data['race_name'].nunique()
            print(f"🏁 Carreras únicas: {races}")
            
        # Información de pilotos
        if 'driver' in self.data.columns:
            drivers = self.data['driver'].nunique()
            print(f"🏎️  Pilotos únicos: {drivers}")
            
        # Valores faltantes
        missing_data = self.data.isnull().sum()
        missing_percentage = (missing_data / len(self.data)) * 100
        columns_with_missing = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(columns_with_missing) > 0:
            print(f"\n⚠️  Columnas con valores faltantes:")
            for col, missing_count in columns_with_missing.head(10).items():
                pct = missing_percentage[col]
                print(f"   {col}: {missing_count:,} ({pct:.1f}%)")
        else:
            print(f"\n✅ Sin valores faltantes")
            
        self.analysis_results['general'] = {
            'shape': self.data.shape,
            'memory_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'years': sorted(self.data['year'].unique()) if 'year' in self.data.columns else [],
            'missing_data': columns_with_missing.to_dict()
        }
    
    def analyze_columns_by_type(self):
        """Analiza columnas por tipo de dato"""
        if self.data is None:
            return
            
        print("\n🔍 ANÁLISIS DE COLUMNAS POR TIPO")
        print("=" * 60)
        
        # Clasificar columnas por tipo
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_cols = self.data.select_dtypes(include=['bool']).columns.tolist()
        
        print(f"🔢 Columnas numéricas ({len(numeric_cols)}):")
        for col in numeric_cols:
            dtype_info = str(self.data[col].dtype)
            unique_vals = self.data[col].nunique()
            print(f"   {col} ({dtype_info}) - {unique_vals:,} valores únicos")
            
        print(f"\n📝 Columnas categóricas ({len(categorical_cols)}):")
        for col in categorical_cols:
            unique_vals = self.data[col].nunique()
            print(f"   {col} - {unique_vals:,} valores únicos")
            
        if datetime_cols:
            print(f"\n📅 Columnas de fecha/hora ({len(datetime_cols)}):")
            for col in datetime_cols:
                print(f"   {col}")
                
        if boolean_cols:
            print(f"\n✅ Columnas booleanas ({len(boolean_cols)}):")
            for col in boolean_cols:
                true_count = self.data[col].sum()
                total = len(self.data)
                print(f"   {col} - {true_count:,}/{total:,} verdaderos ({true_count/total*100:.1f}%)")
        
        self.analysis_results['column_types'] = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols,
            'boolean': boolean_cols
        }
    
    def analyze_weather_data(self):
        """Análisis específico de datos meteorológicos"""
        print("\n🌤️  ANÁLISIS DE DATOS METEOROLÓGICOS")
        print("=" * 60)
        
        weather_cols = [col for col in self.data.columns if any(w in col.lower() 
                       for w in ['weather', 'temp', 'humidity', 'rain', 'wind', 'pressure', 'air_temp', 'track_temp'])]
        
        if not weather_cols:
            print("❌ No se encontraron columnas meteorológicas")
            return
            
        print(f"🌡️  Columnas meteorológicas encontradas ({len(weather_cols)}):")
        for col in weather_cols:
            if col in self.data.columns:
                if self.data[col].dtype in ['object']:
                    unique_vals = self.data[col].value_counts()
                    print(f"   {col}:")
                    for val, count in unique_vals.head(5).items():
                        print(f"      {val}: {count:,} registros")
                else:
                    stats = self.data[col].describe()
                    if len(stats) >= 4:
                        print(f"   {col}: min={self.data[col].min():.2f}, max={self.data[col].max():.2f}, media={self.data[col].mean():.2f}")
        
        # Análisis de lluvia si existe
        rain_cols = [col for col in self.data.columns if 'rain' in col.lower()]
        if rain_cols:
            print(f"\n🌧️  Análisis de lluvia:")
            for col in rain_cols:
                if col in self.data.columns:
                    rain_distribution = self.data[col].value_counts()
                    print(f"   {col}:")
                    for val, count in rain_distribution.items():
                        pct = count / len(self.data) * 100
                        print(f"      {val}: {count:,} ({pct:.1f}%)")
    
    def analyze_performance_metrics(self):
        """Análisis de métricas de rendimiento"""
        print("\n🏁 ANÁLISIS DE MÉTRICAS DE RENDIMIENTO")
        print("=" * 60)
        
        performance_cols = [col for col in self.data.columns if any(p in col.lower() 
                           for p in ['lap_time', 'position', 'points', 'speed', 'sector', 'stint', 'tire'])]
        
        if not performance_cols:
            print("❌ No se encontraron columnas de rendimiento")
            return
            
        print(f"🏎️  Columnas de rendimiento encontradas ({len(performance_cols)}):")
        
        for col in performance_cols:
            if col in self.data.columns:
                if self.data[col].dtype in ['object']:
                    unique_vals = self.data[col].nunique()
                    print(f"   {col}: {unique_vals} valores únicos")
                    if unique_vals <= 10:
                        val_counts = self.data[col].value_counts()
                        print(f"      Distribución: {dict(val_counts.head(5))}")
                else:
                    if self.data[col].dtype in [np.number, 'float64', 'int64']:
                        print(f"   {col}: min={self.data[col].min():.3f}, max={self.data[col].max():.3f}, media={self.data[col].mean():.3f}")
    
    def analyze_by_year(self):
        """Análisis detallado por año"""
        if 'year' not in self.data.columns:
            print("\n❌ No se encontró columna 'year' para análisis temporal")
            return
            
        print("\n📅 ANÁLISIS DETALLADO POR AÑO")
        print("=" * 60)
        
        years = sorted(self.data['year'].unique())
        
        for year in years:
            year_data = self.data[self.data['year'] == year]
            print(f"\n🏆 AÑO {year}")
            print("-" * 40)
            
            # Estadísticas básicas
            print(f"📊 Registros: {len(year_data):,}")
            
            # Carreras en ese año
            if 'race_name' in self.data.columns:
                races = year_data['race_name'].nunique()
                print(f"🏁 Carreras: {races}")
                
            # Pilotos en ese año
            if 'driver' in self.data.columns:
                drivers = year_data['driver'].nunique()
                print(f"🏎️  Pilotos: {drivers}")
                
            # Análisis meteorológico del año
            weather_cols = [col for col in year_data.columns if any(w in col.lower() 
                           for w in ['air_temp', 'track_temp', 'humidity'])]
            
            if weather_cols:
                print(f"🌤️  Condiciones meteorológicas promedio:")
                for col in weather_cols:
                    if col in year_data.columns and year_data[col].dtype in [np.number]:
                        avg_val = year_data[col].mean()
                        print(f"   {col}: {avg_val:.2f}")
                        
            # Análisis de lluvia
            rain_cols = [col for col in year_data.columns if 'rain' in col.lower()]
            if rain_cols:
                for col in rain_cols:
                    if col in year_data.columns:
                        rain_percentage = (year_data[col] == 'Sí').sum() / len(year_data) * 100 if 'Sí' in year_data[col].values else 0
                        print(f"🌧️  Porcentaje con lluvia: {rain_percentage:.1f}%")
                        break
                        
            # Performance metrics del año
            if 'lap_time_seconds' in year_data.columns:
                avg_lap_time = year_data['lap_time_seconds'].mean()
                print(f"⏱️  Tiempo de vuelta promedio: {avg_lap_time:.3f}s")
                
            if 'points' in year_data.columns:
                total_points = year_data['points'].sum()
                print(f"🏆 Total puntos otorgados: {total_points}")
    
    def identify_features_for_training(self):
        """Identifica las mejores características para entrenar modelos"""
        print("\n🤖 CARACTERÍSTICAS RECOMENDADAS PARA ENTRENAMIENTO")
        print("=" * 60)
        
        # Características numéricas útiles
        numeric_features = []
        categorical_features = []
        target_candidates = []
        
        for col in self.data.columns:
            col_lower = col.lower()
            
            # Candidatos para variables objetivo
            if any(target in col_lower for target in ['position', 'points', 'lap_time']):
                if self.data[col].dtype in [np.number]:
                    target_candidates.append(col)
                    
            # Características numéricas útiles
            elif any(feature in col_lower for feature in [
                'temp', 'humidity', 'pressure', 'speed', 'sector', 'stint',
                'tire', 'fuel', 'drs', 'gap', 'interval'
            ]):
                if self.data[col].dtype in [np.number]:
                    numeric_features.append(col)
                    
            # Características categóricas útiles  
            elif any(feature in col_lower for feature in [
                'driver', 'team', 'compound', 'weather', 'rain', 'track'
            ]):
                if self.data[col].dtype == 'object':
                    categorical_features.append(col)
        
        print(f"🎯 Variables objetivo candidatas ({len(target_candidates)}):")
        for target in target_candidates:
            unique_vals = self.data[target].nunique()
            print(f"   {target} - {unique_vals:,} valores únicos")
            
        print(f"\n🔢 Características numéricas útiles ({len(numeric_features)}):")
        for feature in numeric_features:
            missing_pct = (self.data[feature].isnull().sum() / len(self.data)) * 100
            print(f"   {feature} - {missing_pct:.1f}% faltantes")
            
        print(f"\n📝 Características categóricas útiles ({len(categorical_features)}):")
        for feature in categorical_features:
            unique_vals = self.data[feature].nunique()
            missing_pct = (self.data[feature].isnull().sum() / len(self.data)) * 100
            print(f"   {feature} - {unique_vals} categorías, {missing_pct:.1f}% faltantes")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        print(f"✅ Para predicción de posición: usar 'position' como objetivo")
        print(f"✅ Para predicción de puntos: usar 'points' como objetivo")
        print(f"✅ Para predicción de tiempo: usar variables de 'lap_time'")
        print(f"⚠️  Aplicar encoding a variables categóricas antes del entrenamiento")
        print(f"⚠️  Normalizar variables numéricas con diferentes escalas")
        print(f"⚠️  Considerar crear features de interacción (ej: temp * humidity)")
        
        self.analysis_results['features'] = {
            'target_candidates': target_candidates,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
    
    def generate_data_quality_report(self):
        """Genera un reporte de calidad de datos"""
        print("\n📋 REPORTE DE CALIDAD DE DATOS")
        print("=" * 60)
        
        total_rows = len(self.data)
        
        # Completitud de datos
        completeness = {}
        for col in self.data.columns:
            non_null_count = self.data[col].count()
            completeness[col] = (non_null_count / total_rows) * 100
            
        # Columnas con alta completitud (>95%)
        high_quality_cols = [col for col, comp in completeness.items() if comp >= 95]
        print(f"✅ Columnas con alta calidad (≥95% completas): {len(high_quality_cols)}")
        
        # Columnas con baja completitud (<50%)
        low_quality_cols = [col for col, comp in completeness.items() if comp < 50]
        if low_quality_cols:
            print(f"❌ Columnas con baja calidad (<50% completas): {len(low_quality_cols)}")
            for col in low_quality_cols:
                print(f"   {col}: {completeness[col]:.1f}% completa")
        else:
            print(f"✅ No hay columnas con baja calidad")
            
        # Duplicados
        duplicates = self.data.duplicated().sum()
        print(f"\n🔄 Filas duplicadas: {duplicates:,} ({duplicates/total_rows*100:.2f}%)")
        
        # Consistency checks
        print(f"\n🔍 Verificaciones de consistencia:")
        
        # Verificar años válidos
        if 'year' in self.data.columns:
            invalid_years = self.data[(self.data['year'] < 2000) | (self.data['year'] > 2030)]
            print(f"   Años inválidos: {len(invalid_years)} registros")
            
        # Verificar posiciones válidas
        if 'position' in self.data.columns:
            invalid_positions = self.data[(self.data['position'] < 1) | (self.data['position'] > 25)]
            print(f"   Posiciones inválidas: {len(invalid_positions)} registros")
    
    def save_analysis_results(self):
        """Guarda los resultados del análisis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio si no existe
        analysis_dir = os.path.join(self.cache_dir, "data_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Guardar resultados
        results_file = os.path.join(analysis_dir, f"data_exploration_{timestamp}.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(self.analysis_results, f)
        
        # Generar reporte de texto
        report_file = os.path.join(analysis_dir, f"data_exploration_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"📊 REPORTE DE EXPLORACIÓN DE DATOS F1\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            if 'general' in self.analysis_results:
                f.write(f"RESUMEN GENERAL:\n")
                f.write(f"Dimensiones: {self.analysis_results['general']['shape']}\n")
                f.write(f"Memoria: {self.analysis_results['general']['memory_mb']:.2f} MB\n")
                f.write(f"Años: {self.analysis_results['general']['years']}\n\n")
                
            if 'features' in self.analysis_results:
                f.write(f"CARACTERÍSTICAS PARA ENTRENAMIENTO:\n")
                f.write(f"Variables objetivo: {self.analysis_results['features']['target_candidates']}\n")
                f.write(f"Features numéricas: {len(self.analysis_results['features']['numeric_features'])}\n")
                f.write(f"Features categóricas: {len(self.analysis_results['features']['categorical_features'])}\n")
        
        print(f"\n💾 Resultados guardados en:")
        print(f"   📊 Datos: {results_file}")
        print(f"   📄 Reporte: {report_file}")
    
    def run_complete_analysis(self):
        """Ejecuta el análisis completo"""
        print("🏎️  F1 DATA EXPLORER")
        print("🔍 Iniciando análisis completo del dataset...")
        print("=" * 60)
        
        # Cargar datos
        if self.load_all_data() is None:
            return
            
        # Ejecutar todos los análisis
        self.general_overview()
        self.analyze_columns_by_type()
        self.analyze_weather_data()
        self.analyze_performance_metrics()
        self.analyze_by_year()
        self.identify_features_for_training()
        self.generate_data_quality_report()
        
        # Guardar resultados
        self.save_analysis_results()
        
        print(f"\n🎉 ¡ANÁLISIS COMPLETADO!")
        print(f"📊 Dataset analizado: {len(self.data):,} registros")
        print(f"📋 Columnas analizadas: {len(self.data.columns)}")
        print(f"📈 Análisis guardado en: app/models_cache/data_analysis/")

def main():
    explorer = F1DataExplorer()
    explorer.run_complete_analysis()

if __name__ == "__main__":
    main()
