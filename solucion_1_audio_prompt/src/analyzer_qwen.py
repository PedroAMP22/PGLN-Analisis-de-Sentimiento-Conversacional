import os
# Gestión de memoria
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import librosa
import gc
import warnings
import numpy as np
from json_repair import repair_json # ¡Vital para modelos locales!

warnings.filterwarnings("ignore")

from transformers import (
    Qwen2AudioForConditionalGeneration,
    Qwen2AudioProcessor,
    BitsAndBytesConfig
)

# ================== CONFIGURACIÓN ==================

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# CONFIGURACIÓN DEL CHUNKING
CHUNK_SECONDS = 60  # Analizar de 60 en 60 segundos
OVERLAP_SECONDS = 0 # Sin solapamiento para no duplicar eventos en el grafo

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Hardware: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

# ================== PROMPTS ==================

def get_prompt_by_strategy(strategy_id, chunk_index, total_chunks):
    # Contexto temporal para que el modelo sepa dónde está
    time_context = f"This is part {chunk_index + 1} of {total_chunks} of a conversation."
    
    json_structure = """
    JSON STRUCTURE:
    {
      "resumen_segmento": "Summary of this specific segment",
      "nodos": [
        {"id": "SpeakerName", "emocion": "EkmanEmotion"}
      ],
      "aristas": [
        {"origen": "SpeakerName", "destino": "TargetName", "relacion": "Verb"}
      ]
    }
    """
    
    constraints = "Use Paul Ekman's 7 emotions. Identify speakers. Output ONLY JSON."

    if strategy_id == 1:
        return f"Role: Psychologist. {time_context} Analyze this segment. {constraints} {json_structure}"
    if strategy_id == 2:
        return f"{time_context} Think step-by-step about prosody and meaning in this segment. {constraints} {json_structure}"
    return f"{time_context} Divide tasks: Speakers, Emotions, Graph for this segment. {constraints} {json_structure}"

# ================== MAIN ==================

def main():
    gc.collect()
    torch.cuda.empty_cache()

    print(f"📥 Cargando {MODEL_ID} en 8-BITS (Chunking Mode)...")

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    try:
        processor = Qwen2AudioProcessor.from_pretrained(MODEL_ID)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto", 
            quantization_config=quant_config
        )
        print("✅ Modelo cargado.")
    except Exception as e:
        print(f"❌ Error carga: {e}")
        return

    # -------- AUDIO --------
    archivos = [f for f in os.listdir(INPUT_DIR) if f.endswith((".mp3", ".wav", ".ogg"))]
    if not archivos: return print("❌ Sin audios")
    
    print("\n--- SELECCIONA AUDIO ---")
    for i, f in enumerate(archivos): print(f"[{i+1}] {f}")
    idx = int(input("👉 Número: ")) - 1
    ruta_audio = os.path.join(INPUT_DIR, archivos[idx])

    # -------- ESTRATEGIA --------
    print("\n--- ESTRATEGIA ---")
    print("1. Zero-Shot | 2. CoT | 3. LtM")
    strat = int(input("👉 Número: "))

    print("\n👂 Cargando audio completo...")
    full_audio, sr = librosa.load(ruta_audio, sr=processor.feature_extractor.sampling_rate)
    total_duration = len(full_audio) / sr
    print(f"   Duración total: {total_duration:.2f} segundos")

    # -------- BUCLE DE CHUNKING --------
    
    # Estructura final acumulada
    final_json = {
        "resumen_global": f"Analysis of {os.path.basename(ruta_audio)} using Strategy {strat}",
        "timeline": []
    }

    # Calcular trozos
    samples_per_chunk = int(CHUNK_SECONDS * sr)
    total_chunks = int(np.ceil(len(full_audio) / samples_per_chunk))

    for i in range(total_chunks):
        start_sample = i * samples_per_chunk
        end_sample = min((i + 1) * samples_per_chunk, len(full_audio))
        
        # Extraer el trozo de audio
        chunk_audio = full_audio[start_sample:end_sample]
        
        print(f"\n🔄 Procesando Chunk {i+1}/{total_chunks} ({start_sample/sr:.0f}s - {end_sample/sr:.0f}s)...")

        # Prompt específico para este trozo
        prompt = get_prompt_by_strategy(strat, i, total_chunks)

        # Preparar inputs
        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": "p"}, {"type": "text", "text": prompt}]}]
        text_formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        audio_inputs = processor.feature_extractor([chunk_audio], sampling_rate=sr, return_tensors="pt")
        
        # Fix máscara
        if "feature_attention_mask" not in audio_inputs:
            audio_inputs["feature_attention_mask"] = torch.ones_like(audio_inputs["input_features"][:, :, 0], dtype=torch.long)

        text_inputs = processor.tokenizer(text_formatted, return_tensors="pt", padding=True)
        inputs = {**audio_inputs, **text_inputs}
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generar
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False,      # <--- ESTO ES CLAVE: Apaga la creatividad
                repetition_penalty=1.0 # Sin penalización para no liarlo
            )

        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Procesar respuesta del chunk
        if "assistant" in response:
            content = response.split("assistant")[-1]
        else:
            content = response

        try:
            # Usamos json_repair para arreglar errores de sintaxis automáticamente
            chunk_data = json.loads(repair_json(content))
            
            # Convertimos el resultado del chunk en una "Fase" del timeline
            fase_obj = {
                "fase": f"Minute {i+1}",
                "descripcion": chunk_data.get("resumen_segmento", "No summary provided"),
                "nodos": chunk_data.get("nodos", []),
                "aristas": chunk_data.get("aristas", [])
            }
            
            final_json["timeline"].append(fase_obj)
            print("   ✅ Chunk procesado correctamente.")

        except Exception as e:
            print(f"   ⚠️ Error en este chunk (se saltará): {e}")
            print(f"   Contenido crudo: {content[:100]}...")

    # -------- GUARDAR RESULTADO FINAL --------
    print("\n💾 Guardando resultado fusionado...")
    out_path = os.path.join(OUTPUT_DIR, f"qwen_chunked_{archivos[idx].split('.')[0]}_strat{strat}.json")
    
    with open(out_path, "w") as f:
        json.dump(final_json, f, indent=4)
    
    print(f"✅ ¡PROCESO TERMINADO! Archivo: {out_path}")

if __name__ == "__main__":
    main()