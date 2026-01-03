import os
from google import genai
from dotenv import load_dotenv

#cargamos entorno apra apikey
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Falta la API Key")
    exit()

client = genai.Client(api_key=api_key)

print("Consultando modelos disponibles (Nueva Librería)...")

try:
    #listamos los modelos
    for m in client.models.list():
        #imprimimos el id del modelo
        #y su nombre
        print(f" - ID: {m.name}") 
            
except Exception as e:
    print(f"Error: {e}")