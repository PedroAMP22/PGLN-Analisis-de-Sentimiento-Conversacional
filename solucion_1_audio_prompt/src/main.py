import os
import json
from analyzer import analizar_audio_gemini
from visualizer import generar_video_evolutivo

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def seleccionar_archivo():
    archivos = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp3', '.wav', '.ogg'))]
    if not archivos:
        print(f"⚠️  No hay audios en: {INPUT_DIR}")
        return None
    print("\n--- 📂 ARCHIVOS DISPONIBLES ---")
    for i, f in enumerate(archivos): print(f"[{i + 1}] {f}")
    
    while True:
        try:
            idx = int(input("\n👉 Selecciona número: ")) - 1
            if 0 <= idx < len(archivos): return archivos[idx]
        except ValueError: pass

def main():
    print("\n--- 🎬 GENERADOR DE VIDEO DE EMOCIONES (GEMINI 2.5) ---")
    
    archivo = seleccionar_archivo()
    if not archivo: return

    nombre_base = input("👉 Nombre del proyecto (enter para default): ").strip()
    if not nombre_base: nombre_base = archivo.split('.')[0]

    try:
        # 1. Análisis
        datos = analizar_audio_gemini(os.path.join(INPUT_DIR, archivo))
        
        # 2. Guardar JSON
        with open(os.path.join(OUTPUT_DIR, f"{nombre_base}.json"), "w", encoding='utf-8') as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)

        # 3. Guardar Resumen de Texto
        resumen = datos.get("resumen_global", "Sin resumen disponible.")
        with open(os.path.join(OUTPUT_DIR, f"{nombre_base}_resumen.txt"), "w", encoding='utf-8') as f:
            f.write(f"ANÁLISIS DE: {archivo}\n")
            f.write("="*40 + "\n\n")
            f.write(resumen)
        print(f"📝 Resumen guardado en: {nombre_base}_resumen.txt")

        # 4. Generar Video (GIF)
        generar_video_evolutivo(datos, OUTPUT_DIR, nombre_base)
        
        print("\n✨ ¡Proceso completado! Revisa la carpeta 'output'.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()