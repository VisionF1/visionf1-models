#!/usr/bin/env python3
"""
Smart Downloader para FastF1 con manejo automático de rate limits
"""

import subprocess
import time
import sys
import os
import re
from datetime import datetime

class SmartF1Downloader:
    def __init__(self):
        self.max_retries = 10
        self.base_sleep = 30  # segundos base de espera
        self.max_sleep = 300  # máximo 5 minutos de espera
        self.rate_limit_patterns = [
            r"rate limit",
            r"too many requests",
            r"429",
            r"please wait",
            r"slow down",
            r"HTTPError: 429",
            r"requests.exceptions.HTTPError: 429",
            r"ConnectionError",
            r"timeout",
            r"Failed to download"
        ]
        
    def run_download_with_retries(self):
        """Ejecuta python main.py 1 con reintentos inteligentes"""
        
        print("🚀 Iniciando descarga inteligente de datos F1...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        attempt = 1
        consecutive_successes = 0
        
        while attempt <= self.max_retries:
            print(f"\n🔄 Intento #{attempt}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Ejecutando descarga...")
            
            # Ejecutar el comando
            result = self._run_main_download()
            
            if result["success"]:
                consecutive_successes += 1
                print(f"\n✅ Descarga completada exitosamente (código: {result['return_code']})")
                
                if self._is_download_complete(result["output"]):
                    print(f"\n🎉 ¡DESCARGA COMPLETA!")
                    print(f"📊 Total intentos necesarios: {attempt}")
                    print(f"⏱️  Tiempo total aproximado: {self._format_duration(attempt * 60)}")
                    return True
                else:
                    print(f"📝 Descarga parcial detectada, continuando...")
                    # Pausa corta entre descargas exitosas
                    self._smart_sleep(5, "Pausa entre descargas exitosas")
                    
            else:
                consecutive_successes = 0
                rate_limited = self._detect_rate_limit(result["output"])
                
                print(f"\n❌ Error en la descarga (código: {result['return_code']})")
                
                if rate_limited:
                    sleep_time = self._calculate_sleep_time(attempt)
                    print(f"⏳ Rate limit detectado")
                    self._smart_sleep(sleep_time, f"Esperando para evitar rate limit (intento {attempt})")
                else:
                    print(f"❌ Error no relacionado con rate limit:")
                    if result['error']:
                        print(f"   {result['error'][:200]}...")
                    # Pausa corta para otros errores
                    self._smart_sleep(10, "Pausa por error general")
            
            attempt += 1
        
        print(f"\n❌ Falló después de {self.max_retries} intentos")
        return False

    def _run_main_download(self):
        """Ejecuta python main.py 1 y captura la salida mostrando logs en tiempo real"""
        try:
            # Usar el entorno virtual si existe
            python_cmd = self._get_python_command()
            
            print(f"🔧 Ejecutando: {python_cmd} main.py 1")
            print("📋 Salida en tiempo real:")
            print("-" * 50)
            
            # Ejecutar el comando con salida en tiempo real
            process = subprocess.Popen(
                [python_cmd, "main.py", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            
            # Leer línea por línea y mostrar en tiempo real
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    line = line.rstrip()
                    print(f"   {line}")  # Mostrar con indentación
                    output_lines.append(line)
                    
                    # Flush para mostrar inmediatamente
                    sys.stdout.flush()
            
            # Esperar a que termine el proceso
            return_code = process.poll()
            full_output = "\n".join(output_lines)
            
            print("-" * 50)
            print(f"🔚 Proceso terminado con código: {return_code}")
            
            return {
                "success": return_code == 0,
                "output": full_output,
                "error": full_output if return_code != 0 else "",
                "return_code": return_code
            }
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout: El proceso tardó más de 10 minutos")
            return {
                "success": False,
                "output": "",
                "error": "Timeout: El proceso tardó más de 10 minutos",
                "return_code": -1
            }
        except Exception as e:
            print(f"❌ Error ejecutando comando: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1
            }

    def _get_python_command(self):
        """Determina el comando Python correcto a usar"""
        # Verificar si existe el entorno virtual
        venv_python = "/home/tom/Desktop/VisionF1/visionf1-models/env/bin/python"
        if os.path.exists(venv_python):
            return venv_python
        
        # Fallback a python3 o python
        if subprocess.run(["which", "python3"], capture_output=True).returncode == 0:
            return "python3"
        return "python"

    def _detect_rate_limit(self, output):
        """Detecta si el error es por rate limiting"""
        output_lower = output.lower()
        
        for pattern in self.rate_limit_patterns:
            if re.search(pattern.lower(), output_lower):
                return True
        
        return False

    def _is_download_complete(self, output):
        """Determina si la descarga está completa basándose en la salida"""
        completion_indicators = [
            "datos descargados y procesados exitosamente",
            "total carreras procesadas",
            "datos completos descargados",
            "✅ datos completos"
        ]
        
        output_lower = output.lower()
        
        # Buscar indicadores de completitud
        for indicator in completion_indicators:
            if indicator in output_lower:
                return True
        
        # También verificar si no hay errores de descarga
        error_indicators = [
            "no se pudieron obtener datos",
            "error descargando",
            "❌ error",
            "failed to download"
        ]
        
        for error in error_indicators:
            if error in output_lower:
                return False
        
        return False

    def _calculate_sleep_time(self, attempt):
        """Calcula tiempo de espera basado en el intento"""
        # Backoff exponencial con jitter
        sleep_time = min(self.base_sleep * (2 ** (attempt - 1)), self.max_sleep)
        
        # Agregar un poco de variación aleatoria
        import random
        jitter = random.uniform(0.8, 1.2)
        sleep_time = int(sleep_time * jitter)
        
        return sleep_time

    def _smart_sleep(self, seconds, reason):
        """Duerme con contador visual"""
        print(f"\n💤 {reason}")
        
        if seconds <= 10:
            print(f"   Esperando {seconds} segundos...")
            time.sleep(seconds)
        else:
            print(f"   Esperando {seconds} segundos:")
            for remaining in range(seconds, 0, -5):
                print(f"   ⏱️  {remaining:3d}s restantes... ({datetime.now().strftime('%H:%M:%S')})")
                time.sleep(5)
                sys.stdout.flush()  # Asegurar que se muestre inmediatamente
        
        print(f"   ✅ Espera completada\n")

    def _format_duration(self, seconds):
        """Formatea duración en formato legible"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def show_status(self):
        """Muestra el estado actual del cache"""
        print("\n📊 ESTADO ACTUAL DEL CACHE:")
        print("=" * 40)
        
        try:
            python_cmd = self._get_python_command()
            
            # Ejecutar comando para ver info del cache
            process = subprocess.run(
                [python_cmd, "-c", """
from app.data.collectors.fastf1_collector import FastF1Collector
collector = FastF1Collector([])
collector.cache_info()
"""],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if process.returncode == 0:
                print(process.stdout)
            else:
                print("❌ No se pudo obtener información del cache")
                print(process.stderr)
                
        except Exception as e:
            print(f"❌ Error obteniendo estado: {e}")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            downloader = SmartF1Downloader()
            downloader.show_status()
            return
        elif sys.argv[1] == "help":
            print("""
🏎️  Smart F1 Downloader - Uso:

python smart_downloader.py          - Ejecutar descarga inteligente
python smart_downloader.py status   - Ver estado del cache
python smart_downloader.py help     - Mostrar esta ayuda

Características:
✅ Manejo automático de rate limits
✅ Reintentos inteligentes con backoff exponencial  
✅ Monitoreo de progreso en tiempo real
✅ Detección automática de descarga completa
✅ Uso del entorno virtual correcto
""")
            return

    downloader = SmartF1Downloader()
    
    # Mostrar estado inicial
    downloader.show_status()
    
    print("\n" + "="*60)
    print("🏎️  SMART F1 DOWNLOADER")
    print("="*60)
    print("ℹ️  Este script manejará automáticamente:")
    print("   • Rate limits de la API de FastF1")
    print("   • Reintentos con tiempos de espera inteligentes")
    print("   • Detección de descarga completa")
    print("   • Pausas apropiadas entre descargas")
    print("="*60)
    
    input("📋 Presiona ENTER para comenzar la descarga inteligente...")
    
    success = downloader.run_download_with_retries()
    
    if success:
        print("\n🎊 ¡DESCARGA COMPLETADA EXITOSAMENTE!")
        print("\n📈 Ahora puedes ejecutar:")
        print("   python main.py 2  # Para entrenar los modelos")
        print("   python main.py 3  # Para hacer predicciones")
    else:
        print("\n💔 La descarga no se completó")
        print("   Puedes intentar ejecutar el script nuevamente")
        print("   o revisar manualmente con: python main.py 1")

if __name__ == "__main__":
    main()
