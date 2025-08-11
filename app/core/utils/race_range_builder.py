import fastf1
import pandas as pd

class RaceRangeBuilder:
    """Construye rangos de carreras para recolección de datos"""
    
    def build_race_range(self, config):
        """Construye el rango de carreras basado en configuración"""
        race_range = []
        years = config.get("years", [2024])
        max_races_per_year = config.get("max_races_per_year", 24)
        
        print(f"🏗️  Construyendo rango de carreras:")
        print(f"   Años configurados: {years}")
        print(f"   Max carreras por año: {max_races_per_year}")
        
        for year in years:
            print(f"🔍 Detectando carreras para {year}...")
            races = self._get_races_for_year(year, max_races_per_year)
            print(f"   ✅ {len(races)} carreras agregadas para {year}")
            race_range.extend(races)
        
        print(f"📊 Total carreras en el rango: {len(race_range)}")
        
        # Debug: Mostrar distribución por año
        year_distribution = {}
        for race in race_range:
            year = race['year']
            year_distribution[year] = year_distribution.get(year, 0) + 1
        
        print(f"📅 Distribución final por año:")
        for year, count in sorted(year_distribution.items()):
            print(f"   {year}: {count} carreras")
        
        return race_range
    
    def _get_races_for_year(self, year, max_races):
        """Obtiene carreras disponibles para un año específico"""
        try:
            schedule = fastf1.get_event_schedule(year)
            available_races = len(schedule)
            print(f"   📅 {available_races} carreras en calendario de {year}")
            
            # Para años futuros, verificar cuáles han ocurrido
            actual_races = self._count_completed_races(year, available_races, max_races)
            races_to_get = min(actual_races, max_races)
            
            print(f"   ✅ {races_to_get} carreras completadas")
            
            return self._build_race_list(year, races_to_get, schedule)
            
        except Exception as e:
            print(f"❌ Error obteniendo calendario de {year}: {e}")
            return self._fallback_races(year)
    
    def _count_completed_races(self, year, available_races, max_races):
        """Cuenta carreras completadas para años actuales/futuros"""
        from datetime import datetime
        current_year = datetime.now().year
        
        if year < current_year:
            return available_races
        
        print(f"   🔍 Verificando carreras completadas en {year}...")
        completed = 0
        
        for round_num in range(1, min(available_races + 1, max_races + 1)):
            try:
                session = fastf1.get_session(year, round_num, 'R')
                if session.date < pd.Timestamp.now():
                    completed += 1
                else:
                    print(f"   ⏸️  Carrera {round_num} aún no ocurrió")
                    break
            except Exception:
                print(f"   ⏸️  No hay datos para carrera {round_num}")
                break
        
        return completed
    
    def _build_race_list(self, year, races_count, schedule):
        """Construye lista de carreras para el año"""
        races = []
        
        for round_num in range(1, races_count + 1):
            try:
                race_info = schedule[schedule['RoundNumber'] == round_num]
                race_name = race_info.iloc[0]['EventName'] if not race_info.empty else f"Race_{round_num}"
                
                races.append({
                    'year': year,
                    'race_name': race_name,
                    'round_number': round_num
                })
            except Exception as e:
                print(f"   ⚠️  Error con carrera {round_num}: {e}")
                races.append({
                    'year': year,
                    'race_name': f"Race_{round_num}",
                    'round_number': round_num
                })
        
        return races
    
    def _fallback_races(self, year):
        """Fallback para cuando falla la obtención del calendario"""
        from datetime import datetime
        current_year = datetime.now().year
        
        fallback_count = 24 if year <= current_year else 13
        print(f"   🔄 Usando fallback: {fallback_count} carreras")
        
        return [{
            'year': year,
            'race_name': f"Race_{i}",
            'round_number': i
        } for i in range(1, fallback_count + 1)]