"""
Enhanced Data Preparer con Feature Engineering Avanzado
Integra el feature engineering avanzado en el pipeline de preparación de datos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Importar el feature engineer avanzado
from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

class EnhancedDataPreparer:
    def __init__(self, use_advanced_features=True):
        self.use_advanced_features = use_advanced_features
        self.feature_engineer = AdvancedFeatureEngineer() if use_advanced_features else None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def prepare_enhanced_features(self, df):
        """Prepara features usando el pipeline mejorado"""
        print("\n🚀 PREPARANDO FEATURES MEJORADAS")
        print("=" * 50)
        
        original_shape = df.shape
        print(f"📊 Datos originales: {original_shape}")
        
        # 1. Aplicar feature engineering avanzado si está habilitado
        if self.use_advanced_features and self.feature_engineer:
            print(f"\n🔧 Aplicando feature engineering avanzado...")
            df = self.feature_engineer.create_all_advanced_features(df)
            print(f"   ✅ Features avanzadas creadas")
        
        # 2. Mapeo histórico de equipos (como antes)
        print(f"\n🏎️  Aplicando mapeo histórico de equipos...")
        df = self._apply_team_mapping(df)
        
        # 3. Filtrar registros sin equipo válido
        print(f"\n🔄 Filtrando registros sin equipo válido...")
        valid_teams = ['Alpine', 'Aston Martin', 'Ferrari', 'Haas F1 Team', 'Kick Sauber', 
                      'McLaren', 'Mercedes', 'Racing Bulls', 'Red Bull Racing', 'Williams']
        
        before_filter = len(df)
        df = df[df['team'].isin(valid_teams)]
        after_filter = len(df)
        
        if before_filter > after_filter:
            print(f"   🗑️  {before_filter - after_filter} registros eliminados (sin equipo válido)")
        print(f"   ✅ {after_filter} registros restantes con equipos válidos")
        
        # 4. Preparar características base + avanzadas
        print(f"\n🔧 Preparando características finales...")
        
        # Features base esenciales
        base_features = [
            'team_encoded',
            'grid_position',
            'quali_best_time', 
            'race_best_lap_time',
            'clean_air_pace'
        ]
        
        # Features meteorológicas
        weather_features = [
            'session_air_temp',
            'session_track_temp', 
            'session_humidity',
            'session_rainfall'
        ]
        
        # Features de performance histórico del equipo
        historical_features = [
            'team_avg_position_2024',
            'team_avg_position_2023', 
            'team_avg_position_2022'
        ]
        
        # Combinar todas las features disponibles
        all_features = base_features + weather_features + historical_features
        
        # Agregar features avanzadas si están disponibles
        if self.use_advanced_features and self.feature_engineer:
            created_features = [f for f in self.feature_engineer.created_features if f in df.columns]
            # Seleccionar features específicas solicitadas
            priority_advanced_features = [
                f for f in created_features if any(x in f for x in [
                    'quali_gap_to_pole',           # Performance relativo
                    'fp1_gap_to_fastest',          # Performance relativo
                    'team_quali_rank',             # Performance relativo
                    'avg_position_last_3',         # Momentum (con manejo de valores por defecto)
                    'points_last_3',               # Momentum
                    'heat_index',                  # Weather básico
                    'weather_difficulty_index',    # Weather avanzado
                    'team_track_avg_position',     # Compatibilidad circuito
                    'fp1_to_quali_improvement',    # Consistencia
                    'sector_consistency',          # Sectores/velocidad
                    'grid_to_race_change',         # Estratégico
                    'overtaking_ability'           # Estratégico
                ])
            ]
            all_features.extend(priority_advanced_features)
        
        # Filtrar features que existen en el DataFrame
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"   📊 Features base disponibles: {len([f for f in base_features if f in df.columns])}")
        print(f"   🌤️  Features meteorológicas: {len([f for f in weather_features if f in df.columns])}")
        print(f"   📈 Features históricas: {len([f for f in historical_features if f in df.columns])}")
        
        if self.use_advanced_features:
            advanced_count = len([f for f in available_features if f not in base_features + weather_features + historical_features])
            print(f"   🚀 Features avanzadas seleccionadas: {advanced_count}")
        
        # 5. Procesar team encoding
        if 'team' in df.columns:
            df['team_encoded'] = self.label_encoder.fit_transform(df['team'])
            unique_teams_final = df['team'].nunique()
            print(f"   ✅ team_encoded: {unique_teams_final} equipos únicos (sin Unknown)")
        
        # 6. Calcular performance histórico de equipos
        df = self._calculate_team_historical_performance(df)
        
        # 7. Procesar características meteorológicas
        df = self._process_weather_features(df)
        
        # 8. Crear X e y
        X = df[available_features].copy()
        
        # NUEVO: Garantizar que tenemos exactamente 20 features para predicciones
        X = self._ensure_20_features(X)
        
        # 9. Preparar variable objetivo
        target_columns = ['final_position', 'race_position', 'position']
        target_col = None
        y = None
        
        for col in target_columns:
            if col in df.columns:
                target_col = col
                y = df[col].copy()
                break
                
        if y is not None:
            print(f"   ✅ Variable objetivo preparada: {target_col}")
            print(f"   📊 Rango: {y.min()} - {y.max()}")
        else:
            print(f"   ❌ No se encontró ninguna variable objetivo en: {target_columns}")
            print(f"   📊 Columnas disponibles: {[col for col in df.columns if 'position' in col.lower()]}")
            return None, None, None, None
        
        self.feature_names = available_features
        
        print(f"\n📏 Sincronizando datos: X{X.shape}, y{len(y)}")
        
        # 10. Limpiar datos faltantes
        X, y = self._clean_missing_data(X, y)
        
        final_shape = X.shape
        print(f"\n✅ FEATURES MEJORADAS COMPLETADAS")
        print(f"   📊 Shape final: {final_shape}")
        print(f"   🎯 Features totales: {final_shape[1]}")
        print(f"   📈 Incremento features: {final_shape[1] - len(base_features + weather_features + historical_features)}")
        
        return X, y, self.label_encoder, self.feature_names
    
    def _apply_team_mapping(self, df):
        """Aplica mapeo histórico de equipos"""
        team_mapping = {
            'AlphaTauri': 'Racing Bulls',
            'Alpha Tauri': 'Racing Bulls', 
            'RB': 'Racing Bulls',
            'Alfa Romeo': 'Kick Sauber',
            'Sauber': 'Kick Sauber',
            'Aston Martin Aramco Cognizant F1 Team': 'Aston Martin',
            'Mercedes-AMG Petronas F1 Team': 'Mercedes',
            'Oracle Red Bull Racing': 'Red Bull Racing',
            'Red Bull Racing Honda RBPT': 'Red Bull Racing',
            'Scuderia Ferrari': 'Ferrari',
            'McLaren F1 Team': 'McLaren',
            'BWT Alpine F1 Team': 'Alpine',
            'Williams Racing': 'Williams',
            'MoneyGram Haas F1 Team': 'Haas F1 Team',
            # Mapeos adicionales para configuraciones
            'Haas': 'Haas F1 Team',
            'Kick Sauber': 'Kick Sauber'  # Mantener nombre si ya es correcto
        }
        
        if 'team' in df.columns:
            df['team'] = df['team'].replace(team_mapping)
            unique_teams = df['team'].nunique()
            print(f"   ✅ Equipos después del mapeo: {unique_teams} equipos únicos")
            
            # Mostrar equipos válidos
            valid_teams = sorted(df['team'].dropna().unique())
            print(f"   🏎️ Equipos: {', '.join(valid_teams)}")
        
        return df
    
    def _calculate_team_historical_performance(self, df):
        """Calcula performance histórico de equipos"""
        if 'team' in df.columns and 'race_position' in df.columns and 'year' in df.columns:
            print(f"   ✅ Performance histórico calculado para {len(df)} registros con equipo")
            
            # Performance por año
            for year in [2024, 2023, 2022]:
                col_name = f'team_avg_position_{year}'
                year_data = df[df['year'] == year]
                if len(year_data) > 0:
                    team_performance = year_data.groupby('team')['race_position'].mean()
                    df[col_name] = df['team'].map(team_performance).fillna(df['race_position'].mean())
                else:
                    df[col_name] = df['race_position'].mean()
        
        return df
    
    def _process_weather_features(self, df):
        """Procesa características meteorológicas"""
        weather_cols = ['session_air_temp', 'session_track_temp', 'session_humidity', 'session_rainfall']
        
        print(f"   🌤️  Procesando condiciones meteorológicas:")
        
        for col in weather_cols:
            if col in df.columns:
                if col == 'session_rainfall':
                    # Contar sesiones con lluvia
                    rain_sessions = (df[col] > 0).sum() if df[col].dtype in [np.number] else (df[col] == True).sum()
                    total_sessions = len(df)
                    rain_percentage = (rain_sessions / total_sessions) * 100
                    print(f"   ✅ {col}: {rain_sessions} sesiones con lluvia de {total_sessions} ({rain_percentage:.1f}%)")
                else:
                    # Para temperatura y humedad
                    min_val = df[col].min()
                    max_val = df[col].max()
                    unit = '°C' if 'temp' in col else '%' if 'humidity' in col else ''
                    print(f"   ✅ {col}: rango {min_val:.1f}{unit} - {max_val:.1f}{unit}")
        
        return df
    
    def _clean_missing_data(self, X, y):
        """Limpia datos faltantes"""
        print(f"\n🧹 Limpiando datos...")
        
        # Identificar columnas con valores faltantes
        missing_info = X.isnull().sum()
        columns_with_missing = missing_info[missing_info > 0]
        
        if len(columns_with_missing) > 0:
            print(f"   📊 Valores faltantes encontrados:")
            for col, missing_count in columns_with_missing.items():
                percentage = (missing_count / len(X)) * 100
                print(f"      {col}: {missing_count} ({percentage:.1f}%)")
                
                # Estrategias de imputación inteligente
                if 'position' in col:
                    # Para posiciones, usar mediana
                    X[col] = X[col].fillna(X[col].median())
                elif 'temp' in col or 'humidity' in col:
                    # Para datos meteorológicos, usar media
                    X[col] = X[col].fillna(X[col].mean())
                elif 'rainfall' in col:
                    # Para lluvia, asumir sin lluvia
                    X[col] = X[col].fillna(0)
                elif 'time' in col:
                    # Para tiempos, usar mediana
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # Por defecto, usar mediana
                    X[col] = X[col].fillna(X[col].median())
        
        # Verificar que no queden NaN
        remaining_nan = X.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"   ⚠️  {remaining_nan} valores NaN restantes - rellenando con mediana")
            X = X.fillna(X.median())
        
        # Sincronizar X e y (eliminar filas donde y es NaN)
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"   ✅ Datos limpios: {X.shape}")
        
        return X, y
    
    def get_feature_importance_summary(self):
        """Devuelve resumen de importancia de features"""
        if not self.use_advanced_features or not self.feature_engineer:
            return "Feature engineering avanzado no está habilitado"
        
        groups = self.feature_engineer.get_feature_importance_groups()
        summary = "\n🎯 RESUMEN DE FEATURES POR CATEGORÍA:\n"
        
        for category, features in groups.items():
            available_features = [f for f in features if f in self.feature_names]
            summary += f"   {category}: {len(available_features)} features\n"
            
        return summary

    def prepare_training_data(self, df):
        """
        Prepara datos para entrenamiento usando el pipeline mejorado
        Compatible con el pipeline principal
        """
        print("🚀 Preparando datos para entrenamiento con features avanzadas...")
        
        # Usar el método principal de preparación
        result = self.prepare_enhanced_features(df)
        
        # El método devuelve 4 elementos: X, y, label_encoder, feature_names
        if isinstance(result, tuple) and len(result) == 4:
            X, y, label_encoder, feature_names = result
        else:
            print("❌ Error: formato de respuesta inesperado de prepare_enhanced_features")
            return None, None, None, None, None
        
        if X is None:
            print("❌ Error en preparación de features")
            return None, None, None, None, None
        
        if y is None:
            print("❌ No se encontró variable objetivo para entrenamiento")
            return None, None, None, None, None
        
        # Dividir en entrenamiento y prueba (80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"✅ División entrenamiento/prueba completada:")
        print(f"   📊 Entrenamiento: X{X_train.shape}, y{len(y_train)}")
        print(f"   📊 Prueba: X{X_test.shape}, y{len(y_test)}")
        
        return X_train, X_test, y_train, y_test, feature_names
        
        # Dividir datos para entrenamiento y test
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"✅ Split completado:")
        print(f"   📊 Entrenamiento: {len(X_train)} muestras")
        print(f"   📊 Test: {len(X_test)} muestras")
        print(f"   🔧 Features: {X.shape[1]}")
        
        # Retornar en el formato esperado por el pipeline
        return X_train, X_test, y_train, y_test, self.label_encoder
    
    def _ensure_20_features(self, X):
        """Garantiza que X tenga exactamente 20 features, completando las faltantes"""
        
        # Lista exacta de las 20 features usadas en entrenamiento (del feature_names.pkl)
        target_features = [
            'grid_position',
            'quali_best_time',
            'race_best_lap_time',
            'clean_air_pace',
            'session_air_temp',
            'session_track_temp',
            'session_humidity',
            'session_rainfall',
            'quali_gap_to_pole',
            'fp1_gap_to_fastest',
            'team_quali_rank',
            'avg_position_last_3',
            'points_last_3',
            'heat_index',
            'weather_difficulty_index',
            'team_track_avg_position',
            'sector_consistency',
            'fp1_to_quali_improvement',
            'grid_to_race_change',
            'overtaking_ability'
        ]
        
        print(f"   🔧 Asegurando 20 features para compatibilidad con modelos...")
        
        # Crear DataFrame con todas las features necesarias
        result = pd.DataFrame(index=X.index)
        
        for feature in target_features:
            if feature in X.columns:
                result[feature] = X[feature]
            else:
                # Crear feature faltante con valor por defecto inteligente
                if 'position' in feature:
                    default_value = 10.0  # Posición media
                elif 'temp' in feature:
                    default_value = 25.0  # Temperatura media
                elif 'humidity' in feature:
                    default_value = 50.0  # Humedad media
                elif 'rainfall' in feature:
                    default_value = 0.0   # Sin lluvia
                elif 'pace' in feature or 'time' in feature:
                    default_value = 90.0  # Tiempo medio
                elif 'gap' in feature:
                    default_value = 1.0   # Gap medio
                elif 'rank' in feature:
                    default_value = 10.0  # Ranking medio
                elif 'points' in feature:
                    default_value = 5.0   # Puntos medios
                elif 'index' in feature:
                    default_value = 1.0   # Índice neutro
                elif 'consistency' in feature or 'improvement' in feature:
                    default_value = 0.5   # Valor medio
                elif 'change' in feature or 'ability' in feature:
                    default_value = 0.0   # Neutro
                else:
                    default_value = 0.0   # Genérico
                
                result[feature] = default_value
                print(f"     🆕 {feature}: creada con valor {default_value}")
        
        print(f"   ✅ Features finales: {len(result.columns)} (objetivo: 20)")
        
        return result
