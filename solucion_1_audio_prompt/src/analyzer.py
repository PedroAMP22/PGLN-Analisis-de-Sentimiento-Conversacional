import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar entorno
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def analizar_audio_gemini(ruta_audio):
    print(f"📤 Procesando audio: {ruta_audio}...")
    
    # 1. Subir archivo a Google AI (Temporal)
    archivo_nube = genai.upload_file(ruta_audio)
    
    # Esperar a que Google procese el audio
    while archivo_nube.state.name == "PROCESSING":
        print("⏳ Esperando procesamiento en la nube...")
        time.sleep(2)
        archivo_nube = genai.get_file(archivo_nube.name)

    if archivo_nube.state.name == "FAILED":
        raise Exception("❌ Falló el procesamiento del audio en Google.")

    print("✅ Audio listo. Analizando emociones y relaciones...")

    # 2. Configurar Modelo (Gemini 1.5 Flash es Multimodal y Gratis)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # 3. PROMPT ENGINEERING
    # Aquí es donde ocurre la magia. Le pedimos que escuche y estructure.
    prompt = """
    Actúa como un psicólogo experto en dinámicas de grupo y análisis de sentimientos.
    Escucha el archivo de audio proporcionado.
    
    Tu objetivo es generar un grafo de relaciones basado en la conversación.
    
    Instrucciones:
    1. Identifica a los distintos hablantes (puedes usar nombres genéricos como Persona A si no dicen su nombre, pero intenta deducirlos).
    2. Detecta la emoción predominante de cada hablante basándote en su TONO DE VOZ, no solo en las palabras (ej: sarcasmo, gritos, llanto, risa).
    3. Identifica las interacciones: ¿Quién habla a quién? ¿Es una pregunta, un ataque, un apoyo?
    
    Formato de Salida (JSON Estricto):
    Devuelve SOLO un objeto JSON con esta estructura exacta:
    {
      "nodos": [
        {"id": "Nombre", "emocion": "Emoción detectada", "color": "HexCode"}
      ],
      "aristas": [
        {"origen": "Nombre", "destino": "Nombre", "relacion": "Verbo corto", "peso": 1}
      ]
    }
    
    Guía de colores para emociones:
    - Ira/Enojo: #FF4C4C
    - Alegría/Entusiasmo: #FFD700
    - Tristeza/Miedo: #4682B4
    - Neutro/Calma: #A9A9A9
    - Confianza/Apoyo: #32CD32
    """

    # 4. Generar respuesta
    respuesta = model.generate_content(
        [archivo_nube, prompt],
        generation_config={"response_mime_type": "application/json"}
    )

    # 5. Limpieza (Borrar archivo de la nube)
    archivo_nube.delete()

    return json.loads(respuesta.text)