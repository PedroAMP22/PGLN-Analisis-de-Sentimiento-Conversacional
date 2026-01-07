import os
import time
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from json_repair import repair_json

#cargamos entorno para la api-key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: Falta GOOGLE_API_KEY en .env")
    exit()

#cliente
client = genai.Client(api_key=api_key)

#configuracion de paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#modelo al que se hace la peticion
#MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "gemini-3-pro-preview"

def get_prompt_by_strategy(strategy_id):
    #esta es la estructura que ponemos como ejemplo en cada uno de los prompts del JSON
    json_structure = """
    JSON STRUCTURE:
    {
      "resumen_global": "Brief summary of the situation.",
      "timeline": [
        {
          "fase": "Phase Name (e.g., Start, Conflict)",
          "descripcion": "Context description",
          "nodos": [
            {"id": "SpeakerName", "emocion": "EkmanEmotion"}
          ],
          "aristas": [
            {"origen": "SpeakerName", "destino": "TargetName", "relacion": "Verb (e.g., Blames, Asks)"}
          ]
        }
      ]
    }
    """

    #zero-shot + role
    if strategy_id == 1:
        return f"""
        ROLE: Act as an expert Psychologist and Data Scientist specializing in social dynamics.

        TASK: Analyze the attached audio file. Identify the speakers, classify their emotions using ONLY Paul Ekman's 7 basic emotions (Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger), and map the directional interactions between them.

        FORMAT: You must output the result strictly as a valid JSON object. Do not add markdown text before or after, just the JSON code block.

        {json_structure}
        """

    #CoT
    if strategy_id == 2:
        return f"""
        Analyze the attached audio following a step-by-step reasoning process (Chain of Thought).

        STEP 1: PROSODY ANALYSIS. Listen to the tone, pitch, and speed. Is the voice shaky? Loud?
        STEP 2: SEMANTIC ANALYSIS. What are they actually saying? Does it match the tone?
        STEP 3: INTERACTION MAPPING. Who initiates? Who responds?

        FINAL STEP: Generate the JSON output based on your reasoning.

        CONSTRAINTS:
        - Use Paul Ekman's 7 emotions: [Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger].
        - Output ONLY the JSON object at the end.

        {json_structure}
        """

    #Least-to-Most
    return f"""
    You will solve this analysis by breaking it down into sub-problems.

    Sub-problem 1: Identify the list of unique speakers in the audio.
    Sub-problem 2: Segment the audio into logical temporal phases.
    Sub-problem 3: For each phase, determine the emotion of each speaker (Use Ekman's 7 emotions).
    Sub-problem 4: For each phase, determine the directed edges (interactions) between speakers.

    Main Problem: Compile all sub-problem solutions into a strict JSON format.

    OUTPUT: Provide the final JSON object.

    {json_structure}
    """

def main():    
    #seleccion del audio local
    if not os.path.exists(INPUT_DIR): return print(f"No existe: {INPUT_DIR}")
    archivos = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp3', '.wav', '.ogg', '.m4a'))]
    if not archivos: return print("Sin audios")
        
    for i, f in enumerate(archivos): print(f"[{i+1}] {f}")
    
    try:
        idx = int(input("Audio: ")) - 1
        ruta_audio = os.path.join(INPUT_DIR, archivos[idx])
    except: return print("Selección inválida.")

    #menu para seleccionar el prompt con el que queremos probar permite elegir entre las 3 opciones
    print("\n1. Zero-Shot (R-T-F) | 2. CoT | 3. LtM")
    try:
        strat = int(input("Estrategia: "))
        prompt = get_prompt_by_strategy(strat)
    except: return

    #procesar la peticion
    print(f"\nSubiendo a Google AI (Nueva API)...")
    try:
        #primero subimos el archivo con el cliente
        archivo_nube = client.files.upload(file=ruta_audio)
        
        print("Procesando audio...", end="", flush=True)
        while archivo_nube.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            archivo_nube = client.files.get(name=archivo_nube.name)
        print("Listo.")

        if archivo_nube.state.name == "FAILED":
            raise Exception("Google falló al procesar el audio.")

        print(f"\nAnalizando con {MODEL_NAME}...")
        
        #generamos el contenido
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=archivo_nube.uri,
                            mime_type=archivo_nube.mime_type),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", #tipo de respuesta
                temperature=0.2 #temperatura del modelo
            )
        )
        
        #guardamos y reparamos en caso de json mal formateados
        data = json.loads(repair_json(response.text))
        #establecemos el nombre del json de salida generado
        nombre_clean = archivos[idx].split('.')[0].replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"gemini3_{nombre_clean}_strat{strat}.json")
        
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Guardado: {out_path}")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        #limpiamos aunque gogle ya lo borra a los 2 dias
        if 'archivo_nube' in locals():
            try:
                client.files.delete(name=archivo_nube.name)
            except: pass

if __name__ == "__main__":
    main()

"""
Gemini fue el que mejores resultados obtuvo y esto es el resumen de los diferentes prompts.
- Zero-shot and role: Al guiar al modelo al espacio latente relacionado con los sentimientos mediante el role pudimos realizar un
ánalisis de los audios preciso salvo en caso de emociones como sarcasmo que analizaba de manera superficial debido a la ausencia de
un razonamiento forzado sobre los pasos para realizar un ánalisis profundo.
- CoT: Este prompt mejoró la identificación de emociones o relaciones más complejas como puede ser el sarcasmos sin embargo, al especificarle
diferentes niveles de razonamiento el modelo alucinaba en ciertos audios donde participaban muchas personas como the office con diferentes timbres
y se volvía demasiado estricto llegando a separar a una misma persona en 2 diferentes.
- LTM: esta sin duda fue la mejor solución al permitirle razonar y descomponer el problema en subproblemas más sencillos que se identificaban
con los pasos lógicos para esta tarea. Como puede ser identificar primero a los hablantes que eran los "nodos" antes de establcer las "aristas" o relaciones
entre estos.
Para el caso de GPT el mayor limitante fue el uso de whisper que utiliza para transformarlo a texto que sin duda es un gran cuello de botella
y elimina información clave como puede ser el tono de voz.
"""