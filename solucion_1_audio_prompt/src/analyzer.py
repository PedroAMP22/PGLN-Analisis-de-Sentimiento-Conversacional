import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def analizar_audio_gemini(ruta_audio):
    print(f"📤 Procesando audio: {ruta_audio}...")
    
    archivo_nube = genai.upload_file(ruta_audio)
    
    while archivo_nube.state.name == "PROCESSING":
        time.sleep(2)
        archivo_nube = genai.get_file(archivo_nube.name)

    if archivo_nube.state.name == "FAILED":
        raise Exception("❌ Falló el procesamiento en Google.")

    print("✅ Audio listo. Analizando...")

    #gemini-3-pro-preview gemini-2.5-flash
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
    Adopt a **Multi-Expert Persona** strategy to analyze this audio file. You must act as three experts simultaneously:

    1. **The Psycholinguist:** Analyze the prosody (tone, pitch, speed) and lexical content to detect subtle emotional cues.
    2. **The Social Network Analyst:** Identify directional interactions (who speaks to whom) and the nature of the tie.
    3. **The Data Engineer:** Ensure the output is strictly formatted JSON without syntax errors.

    ### CONSTRAINTS (Strict Adherence Required):
    - **Emotion Classification:** You MUST classify emotions ONLY into Paul Ekman's 7 Basic Emotions:
      ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger'].
    - **Timeline:** Segment the conversation into logical phases (e.g., Start, Conflict, Resolution).
    - **Cast Consistency:** Identify ALL participants. If a participant is silent in a phase, they must still appear in the node list (implied presence).

    ### OUTPUT FORMAT (JSON ONLY):
    {
      "resumen_global": "A brief psycholinguistic summary of the interaction.",
      "timeline": [
        {
          "fase": "Phase Name",
          "descripcion": "Contextual description",
          "nodos": [
            {"id": "SpeakerName", "emocion": "One of the 7 Ekman emotions"}
          ],
          "aristas": [
            {"origen": "SpeakerName", "destino": "TargetName", "relacion": "Verb (e.g., Blames, Asks)", "peso": 1-10}
          ]
        }
      ]
    }
    """

    # Reintentos automáticos
    for i in range(3):
        try:
            respuesta = model.generate_content(
                [archivo_nube, prompt],
                generation_config={"response_mime_type": "application/json"}
            )
            archivo_nube.delete()
            return json.loads(respuesta.text)
        except Exception as e:
            if "429" in str(e):
                print("⏳ Cuota llena, esperando 30s...")
                time.sleep(30)
            else:
                archivo_nube.delete()
                raise e
    
    archivo_nube.delete()
    raise Exception("❌ Error persistente de API.")