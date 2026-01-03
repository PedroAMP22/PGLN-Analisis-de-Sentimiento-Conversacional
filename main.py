import streamlit as st
import time
import os
import pandas as pd

# =========================================================
# BLOQUE DE FUNCIONES - SUSTITUYE AQUÍ TU LÓGICA
# =========================================================

def run_audio_fine_tuning(): # pedro
    """Llamada a tu solución 1: Audio + Fine-Tuning (WavLM)"""
    time.sleep(5) # Simulación
    return {"gif": "solucion_2_audio_finetuning/social_emotion_graph.gif", "json" : "solucion_2_audio_finetuning/dynamic_scene_analysis.json"}

def run_audio_prompt_engineering(): # alvaro
    """Llamada a tu solución 2: Audio + Prompt Engineering (Whisper + GPT/Claude)"""
    time.sleep(5) # Simulación
    return {"gif": "solucion_1_audio_prompt/output/gemini_3_marriage_ltm.gif", "json" : "solucion_1_audio_prompt\output\gemini_3_marriage_ltm.json"}

def run_text_fine_tuning(): # chen
    """Llamada a tu solución 3: Texto + Fine-Tuning (BERT/RoBERTa)"""
    time.sleep(3) # Simulación
    return {"accuracy": "91%", "time": "1.5s", "gif": "social_emotion_graph.gif"}

def run_text_prompt_engineering(): # david
    """Llamada a tu solución 4: Texto + Prompt Engineering (GPT-4)"""
    time.sleep(3) # Simulación
    return {"accuracy": "85%", "time": "2.2s", "gif": "social_emotion_graph.gif"}

# =========================================================
# CONFIGURACIÓN DE LA INTERFAZ
# =========================================================

st.set_page_config(page_title="SentimAI Multimodal", page_icon="🎭", layout="centered")

# Estilos CSS para mejorar la apariencia
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; font-weight: bold; }
    .main-title { text-align: center; color: #1E88E5; font-family: 'Helvetica'; }
    .step-text { color: #555; font-size: 1.2em; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# Inicializar estados de la sesión
if 'step' not in st.session_state: st.session_state.step = 1
if 'data_type' not in st.session_state: st.session_state.data_type = None
if 'method' not in st.session_state: st.session_state.method = None
if 'results' not in st.session_state: st.session_state.results = None

def go_to_step(n): st.session_state.step = n

def restart():
    st.session_state.step = 1
    st.session_state.data_type = None
    st.session_state.method = None
    st.session_state.results = None

# --- PANTALLA 1: MODALIDAD ---
if st.session_state.step == 1:
    st.markdown("<h1 class='main-title'>🎭 SentimAI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='step-text'>Paso 1: Selecciona la modalidad del dato</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 AUDIO\n(Señales Acústicas)"):
            st.session_state.data_type = "Audio"
            go_to_step(2)
            st.rerun()
    with col2:
        if st.button("📄 TEXTO\n(Contenido Lingüístico)"):
            st.session_state.data_type = "Texto"
            go_to_step(2)
            st.rerun()

# --- PANTALLA 2: METODOLOGÍA ---
elif st.session_state.step == 2:
    st.markdown(f"<h1 class='main-title'>Análisis de {st.session_state.data_type}</h1>", unsafe_allow_html=True)
    st.markdown("<p class='step-text'>Paso 2: Selecciona la metodología de análisis</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 FINE-TUNING\n(Modelos Especializados)"):
            st.session_state.method = "Fine-Tuning"
            go_to_step(3)
            st.rerun()
    with col2:
        if st.button("✍️ PROMPT ENGINEERING\n(Modelos Fundacionales)"):
            st.session_state.method = "Prompt Engineering"
            go_to_step(3)
            st.rerun()
    
    if st.button("⬅️ Volver atrás"):
        go_to_step(1)
        st.rerun()

# --- PANTALLA 3: EJECUCIÓN ---
elif st.session_state.step == 3:
    st.markdown("<h1 class='main-title'>Procesando Escena...</h1>", unsafe_allow_html=True)
    st.info(f"Configuración: **{st.session_state.data_type}** + **{st.session_state.method}**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Simulación de carga visual
    stages = ["Cargando modelos...", "Analizando señales...", "Extrayendo entidades...", "Generando grafo emocional..."]
    for i, stage in enumerate(stages):
        status_text.text(f"⏳ {stage}")
        time.sleep(0.8)
        progress_bar.progress((i + 1) * 20)
    
    # 2. EJECUCIÓN REAL DE LA SOLUCIÓN SELECCIONADA
    try:
        if st.session_state.data_type == "Audio":
            if st.session_state.method == "Fine-Tuning":
                st.session_state.results = run_audio_fine_tuning()
            else:
                st.session_state.results = run_audio_prompt_engineering()
        else:
            if st.session_state.method == "Fine-Tuning":
                st.session_state.results = run_text_fine_tuning()
            else:
                st.session_state.results = run_text_prompt_engineering()
        
        progress_bar.progress(100)
        status_text.text("✅ ¡Completado!")
        time.sleep(1)
        go_to_step(4)
        st.rerun()
    except Exception as e:
        st.error(f"Error ejecutando la solución: {e}")
        if st.button("Reiniciar"): restart(); st.rerun()

# --- PANTALLA 4: RESULTADOS ---
elif st.session_state.step == 4:
    st.markdown("<h1 class='main-title'>📊 Resultados del Análisis</h1>", unsafe_allow_html=True)
    
    res = st.session_state.results # Aquí recibes el diccionario con las rutas
    
    # 1. MOSTRAR EL GIF
    if os.path.exists(res['gif']):
        st.image(res['gif'], caption="Grafo de Interacciones Emocionales", use_container_width=True)
    else:
        st.error(f"No se encontró el GIF en: {res['gif']}")

    st.divider()

    # 2. PREPARAR LA DESCARGA DEL JSON
    if os.path.exists(res['json']):
        # Leemos el contenido del archivo físico
        with open(res['json'], "r", encoding="utf-8") as f:
            json_string = f.read()
        
        # Extraemos solo el nombre del archivo (sin las carpetas) para el nombre de descarga
        nombre_archivo = os.path.basename(res['json'])

        st.download_button(
            label="📥 Descargar Informe JSON",
            data=json_string,
            file_name=nombre_archivo,
            mime="application/json"
        )
    else:
        st.warning("El archivo JSON de resultados no existe en la ruta especificada.")

    # 3. MÉTRICAS (puedes personalizarlas según el JSON si quieres)
    m1, m2, m3 = st.columns(3)
    m1.metric("Modelo", "Multimodal")
    m2.metric("Estado", "Finalizado")
    m3.metric("Archivo", "JSON Generado")

    if st.button("🔄 Nuevo Análisis"):
        restart()
        st.rerun()