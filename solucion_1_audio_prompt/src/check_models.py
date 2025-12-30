import os
from google import genai
from dotenv import load_dotenv

# Cargar entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Falta la API Key")
    exit()

client = genai.Client(api_key=api_key)

print("🔍 Consultando modelos disponibles (Nueva Librería)...")

try:
    # Listar modelos sin filtrar atributos complejos
    for m in client.models.list():
        # Imprimimos el ID del modelo (lo que necesitas poner en el código)
        # y su nombre legible
        print(f" - ID: {m.name}") 
            
except Exception as e:
    print(f"❌ Error: {e}")