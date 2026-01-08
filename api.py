import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import os
import EDA_pipe  # Importa o EDA para ter as funções to_list e split_semicolon

#uvicorn api:app --reload

app = FastAPI(title="Steam Price Predictor", description="API Inteligente")

# --- CARREGAMENTO DO MODELO ---
arquivo_modelo = 'steam_price_model.joblib'
if os.path.exists(arquivo_modelo):
    modelo = load(arquivo_modelo)
    print("Modelo carregado!")
else:
    modelo = None
    print("ERRO: Modelo não encontrado.")

class JogoInput(BaseModel):
    genres: str
    publisher: str
    release_year: int

@app.post("/predict")
def prever_preco(jogo: JogoInput):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo offline.")

    try:
        # --- 1. DEBUG: Ver o que chegou ---
        print(f"\n[API] Recebi: {jogo.dict()}")

        # --- 2. LIMPEZA (CRUCIAL!) ---
        # No EDA.py você usou .str.lower().str.strip() no publisher.
        # Precisamos fazer igual aqui, senão o modelo não reconhece!
        publisher_limpa = jogo.publisher.lower().strip()
        
        # O genero não precisa de lower() pq o CountVectorizer já faz isso, 
        # mas removemos espaços extras por segurança.
        generos_limpos = jogo.genres.strip()

        # --- 3. Montar DataFrame ---
        dados_dict = {
            'genres': [generos_limpos],
            'publisher': [publisher_limpa], 
            'release_year': [jogo.release_year]
        }
        df_novo = pd.DataFrame(dados_dict)

        # --- 4. DEBUG: Ver o que vai para o modelo ---
        print("[API] DataFrame Enviado pro Modelo:")
        print(df_novo)

        # --- 5. Previsão ---
        preco_estimado = modelo.predict(df_novo)[0]
        
        print(f"[API] Resultado: {preco_estimado}")

        return {
            "status": "sucesso",
            "preco_estimado": round(preco_estimado, 2),
            "moeda": "EUR"
        }

    except Exception as e:
        print(f"❌ ERRO NA API: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"status": "online"}