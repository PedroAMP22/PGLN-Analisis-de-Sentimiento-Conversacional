import os
import json
from analyzer import analizar_audio_gemini
from visualizer import generar_grafo_visual

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def seleccionar_archivo():
    """Muestra un menú para elegir el audio."""
    archivos = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp3', '.wav', '.m4a', '.ogg'))]
    
    if not archivos:
        print(f"⚠️  No hay audios en: {INPUT_DIR}")
        return None

    print("\n--- 📂 ARCHIVOS DISPONIBLES ---")
    for i, archivo in enumerate(archivos):
        print(f"[{i + 1}] {archivo}")
    
    while True:
        try:
            seleccion = input("\n👉 Selecciona el número del archivo: ")
            idx = int(seleccion) - 1
            if 0 <= idx < len(archivos):
                return archivos[idx]
            else:
                print("❌ Número inválido. Intenta de nuevo.")
        except ValueError:
            print("❌ Por favor, introduce un número.")

def main():
    print("\n--- 🤖 SISTEMA DE ANÁLISIS DE EMOCIONES (GEMINI 2.5) ---")
    
    # 1. Selección de archivo
    archivo_nombre = seleccionar_archivo()
    if not archivo_nombre:
        return

    ruta_audio = os.path.join(INPUT_DIR, archivo_nombre)

    # 2. Selección de nombre de salida
    nombre_salida = input("👉 Escribe el nombre para guardar los resultados (sin extensión): ").strip()
    if not nombre_salida:
        nombre_salida = "resultado_" + archivo_nombre.split('.')[0] # Default si lo deja vacío
    
    try:
        # 3. Analizar
        print(f"\n🚀 Procesando '{archivo_nombre}' con Gemini...")
        datos_grafo = analizar_audio_gemini(ruta_audio)
        
        # 4. Guardar JSON
        ruta_json = os.path.join(OUTPUT_DIR, f"{nombre_salida}.json")
        with open(ruta_json, "w", encoding='utf-8') as f:
            json.dump(datos_grafo, f, indent=4, ensure_ascii=False)
        print(f"📄 JSON guardado: {ruta_json}")

        # 5. Visualizar (Pasamos el nombre personalizado)
        generar_grafo_visual(datos_grafo, OUTPUT_DIR, nombre_salida)
        
        print("\n✅ ¡Todo listo! Revisa la carpeta 'output'.")

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")

if __name__ == "__main__":
    main()