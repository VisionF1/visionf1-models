from app.config import DRIVERS_2025, ADAPTATION_SYSTEM, PREDICTION_CONFIG

class ProgressiveAdapter:
    def __init__(self):
        self.adaptation_config = ADAPTATION_SYSTEM
        self.drivers_config = DRIVERS_2025
        
    def calculate_adaptation_penalty(self, driver, current_race_number):
        """Calcula la penalización de adaptación basada en el número de carrera actual"""
        driver_info = self.drivers_config.get(driver, {})
        
        # Si el piloto no tiene cambios, no hay penalización
        if not (driver_info.get("rookie", False) or driver_info.get("team_change", False)):
            return 0.0
        
        # Determinar tipo de cambio
        if driver_info.get("rookie", False):
            change_type = "rookie"
        elif driver_info.get("team_change", False):
            change_type = "team_change"
        else:
            return 0.0
        
        # Obtener configuración del tipo de cambio
        change_config = self.adaptation_config["change_types"][change_type]
        base_penalty = change_config["base_penalty"]
        adaptation_races = change_config["adaptation_races"]
        
        # Si ya pasaron las carreras de adaptación, no hay penalización
        if current_race_number > adaptation_races:
            return 0.0
        
        # Calcular penalización progresiva
        progress_ratio = (current_race_number - 1) / adaptation_races
        remaining_penalty = base_penalty * (1 - progress_ratio)
        
        return max(0.0, remaining_penalty)
    
    def get_adaptation_status(self, driver, current_race_number):
        """Obtiene el estado de adaptación de un piloto"""
        driver_info = self.drivers_config.get(driver, {})
        
        if not (driver_info.get("rookie", False) or driver_info.get("team_change", False)):
            return {
                "status": "fully_adapted",
                "penalty": 0.0,
                "description": "Sin cambios - completamente adaptado",
                "progress": 100
            }
        
        # Determinar tipo de cambio
        if driver_info.get("rookie", False):
            change_type = "rookie"
        elif driver_info.get("team_change", False):
            change_type = "team_change"
        else:
            change_type = "unknown"
        
        change_config = self.adaptation_config["change_types"][change_type]
        adaptation_races = change_config["adaptation_races"]
        penalty = self.calculate_adaptation_penalty(driver, current_race_number)
        
        if current_race_number > adaptation_races:
            status = "fully_adapted"
            progress = 100
        else:
            status = "adapting"
            progress = min(100, int((current_race_number / adaptation_races) * 100))
        
        return {
            "status": status,
            "penalty": penalty,
            "description": change_config["description"],
            "progress": progress,
            "races_remaining": max(0, adaptation_races - current_race_number + 1)
        }
    
    def apply_progressive_penalties(self, predictions_df, current_race_number):
        """Aplica penalizaciones progresivas a las predicciones"""
        print(f"📊 Aplicando adaptación progresiva para la carrera #{current_race_number}")
        
        adaptations_applied = []
        
        for idx, row in predictions_df.iterrows():
            driver = row['driver']
            adaptation_status = self.get_adaptation_status(driver, current_race_number)
            penalty = adaptation_status["penalty"]
            
            if penalty > 0:
                # Aplicar penalización a la posición predicha
                old_position = row['predicted_position']
                new_position = old_position + penalty
                predictions_df.loc[idx, 'predicted_position'] = new_position
                
                # Agregar información de adaptación
                predictions_df.loc[idx, 'adaptation_penalty'] = penalty
                predictions_df.loc[idx, 'adaptation_progress'] = adaptation_status["progress"]
                
                adaptations_applied.append({
                    'driver': driver,
                    'old_position': old_position,
                    'new_position': new_position,
                    'penalty': penalty,
                    'status': adaptation_status["status"],
                    'progress': adaptation_status["progress"],
                    'description': adaptation_status["description"]
                })
        
        # Mostrar resumen de adaptaciones
        self._show_adaptation_summary(adaptations_applied, current_race_number)
        
        return predictions_df
    
    def _show_adaptation_summary(self, adaptations, current_race_number):
        """Muestra un resumen de las adaptaciones aplicadas"""
        if not adaptations:
            print("   ✅ Todos los pilotos ya están completamente adaptados")
            return
        
        print(f"\n🔄 ADAPTACIONES APLICADAS - CARRERA #{current_race_number}")
        print("-" * 70)
        print(f"{'Piloto':<6} {'Antes':<6} {'Después':<8} {'Penaliz.':<8} {'Progreso':<8} {'Tipo'}")
        print("-" * 70)
        
        for adaptation in adaptations:
            print(f"{adaptation['driver']:<6} "
                  f"P{adaptation['old_position']:<5.1f} "
                  f"P{adaptation['new_position']:<7.1f} "
                  f"{adaptation['penalty']:<7.1f} "
                  f"{adaptation['progress']:<7d}% "
                  f"{adaptation['description']}")
        
        print(f"\n💡 Las penalizaciones disminuyen automáticamente cada carrera")
        print(f"   Rookies: se adaptan completamente en 8 carreras")
        print(f"   Cambios de equipo: se adaptan completamente en 5 carreras")