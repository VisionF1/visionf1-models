"""
Imputer robusto personalizado para manejar valores faltantes en datos de F1
"""
import pandas as pd
import numpy as np
from typing import Union, Dict, Any

class RobustF1Imputer:
    """Imputer especializado para datos de F1 con estrategias específicas por columna"""
    
    def __init__(self):
        self.imputation_values_ = {}
        self.fitted_ = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'RobustF1Imputer':
        """Aprende estrategias de imputación específicas por columna"""
        print("🔧 Aprendiendo estrategias de imputación...")
        
        self.imputation_values_ = {}
        
        for column in X.columns:
            strategy = self._get_column_strategy(column)
            
            if strategy == 'median':
                value = X[column].median()
                if pd.isnull(value):
                    value = self._get_fallback_value(column)
                    
            elif strategy == 'mode':
                mode_result = X[column].mode()
                value = mode_result.iloc[0] if len(mode_result) > 0 else self._get_fallback_value(column)
                
            elif strategy == 'domain_specific':
                value = self._get_domain_specific_value(column, X)
                
            else:  # 'constant'
                value = strategy['value']
            
            self.imputation_values_[column] = value
            
            # Log para debugging
            missing_count = X[column].isnull().sum()
            if missing_count > 0:
                print(f"📊 {column}: {missing_count} valores faltantes → {value:.3f}")
        
        self.fitted_ = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica las estrategias de imputación aprendidas"""
        if not self.fitted_:
            raise ValueError("Imputer debe ser fit antes de transform")
            
        X_imputed = X.copy()
        
        for column in X_imputed.columns:
            if column in self.imputation_values_:
                missing_mask = X_imputed[column].isnull()
                if missing_mask.any():
                    X_imputed.loc[missing_mask, column] = self.imputation_values_[column]
            else:
                # Nueva columna no vista durante fit
                missing_mask = X_imputed[column].isnull()
                if missing_mask.any():
                    fallback = self._get_fallback_value(column)
                    X_imputed.loc[missing_mask, column] = fallback
                    print(f"⚠️  Nueva columna {column}: usando valor fallback {fallback}")
        
        return X_imputed
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit y transform en una operación"""
        return self.fit(X, y).transform(X)
        
    def _get_column_strategy(self, column: str) -> Union[str, Dict[str, Any]]:
        """Determina la estrategia de imputación por tipo de columna"""
        column_lower = column.lower()
        
        # Estrategias específicas por tipo de feature
        if 'driver' in column_lower and 'encoded' in column_lower:
            return 'mode'  # Para driver_encoded usar moda
        
        elif any(x in column_lower for x in ['sector', 'time', 'pace', 'lap']):
            return 'domain_specific'  # Para tiempos usar estrategia de dominio
            
        elif any(x in column_lower for x in ['position', 'grid', 'quali']):
            return 'median'  # Para posiciones usar mediana
            
        elif any(x in column_lower for x in ['speed', 'gap', 'interval']):
            return 'domain_specific'  # Para velocidades usar estrategia de dominio
            
        else:
            return 'median'  # Por defecto mediana
    
    def _get_domain_specific_value(self, column: str, X: pd.DataFrame) -> float:
        """Calcula valores específicos del dominio F1"""
        column_lower = column.lower()
        
        # Para tiempos de sector (típicamente 20-35 segundos)
        if 'sector' in column_lower and 'time' in column_lower:
            median_val = X[column].median()
            if not pd.isnull(median_val):
                return median_val
            # Fallback basado en sector
            if 'sector1' in column_lower:
                return 24.5  # Sector 1 típico
            elif 'sector2' in column_lower:
                return 30.2  # Sector 2 típico
            else:  # sector3
                return 22.8  # Sector 3 típico
        
        # Para lap times completos (típicamente 75-85 segundos)
        elif any(x in column_lower for x in ['lap', 'time', 'pace']) and 'sector' not in column_lower:
            median_val = X[column].median()
            if not pd.isnull(median_val):
                return median_val
            return 79.5  # Lap time típico
        
        # Para velocidades (típicamente 200-350 km/h)
        elif 'speed' in column_lower:
            median_val = X[column].median()
            if not pd.isnull(median_val):
                return median_val
            return 275.0  # Velocidad típica
        
        # Para gaps/intervals (segundos)
        elif any(x in column_lower for x in ['gap', 'interval']):
            median_val = X[column].median()
            if not pd.isnull(median_val):
                return median_val
            return 1.5  # Gap típico
        
        else:
            # Fallback a mediana
            median_val = X[column].median()
            return median_val if not pd.isnull(median_val) else 0.0
    
    def _get_fallback_value(self, column: str) -> float:
        """Valores de fallback cuando no hay datos válidos"""
        column_lower = column.lower()
        
        if 'driver' in column_lower and 'encoded' in column_lower:
            return 0  # Primer driver
        elif 'position' in column_lower:
            return 10  # Posición media
        elif any(x in column_lower for x in ['sector', 'time', 'pace', 'lap']):
            return 79.5  # Tiempo típico
        elif 'speed' in column_lower:
            return 275.0  # Velocidad típica
        else:
            return 0.0
    
    def get_imputation_report(self) -> pd.DataFrame:
        """Genera reporte de valores de imputación"""
        if not self.fitted_:
            return pd.DataFrame()
            
        report_data = []
        for column, value in self.imputation_values_.items():
            strategy = self._get_column_strategy(column)
            report_data.append({
                'column': column,
                'strategy': strategy,
                'imputation_value': value
            })
        
        return pd.DataFrame(report_data)
