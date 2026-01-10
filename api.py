import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import os

import EDA_pipe 

#uvicorn api:app --reload

app = FastAPI(title="Steam Price Predictor")

try:
    modelo = load('steam_price_model.joblib')
    print("Modelo carregado com sucesso!")
except Exception as e:
    modelo = None
    print(f"Não foi possível carregar o modelo. Detalhes: {e}")

# Definição dos dados de entrada
class JogoInput(BaseModel):
    """
    Define o esquema de dados esperado para a requisição de previsão.
    Utiliza Pydantic para validação automática de tipos.
    """

    genres: str
    categories: str
    publisher: str
    release_year: int



@app.post("/predict")
def prever_preco(jogo: JogoInput):
    """
    Endpoint principal para previsão de preços.
    
    Recebe um objeto JSON com os dados do jogo, converte para DataFrame
    e passa pelo Pipeline de Machine Learning carregado.
    
    Args:
        jogo (JogoInput): Objeto contendo generos, categorias, publisher e ano.
        
    Returns:
        dict: JSON contendo o status, preço estimado e moeda.
    
    Raises:
        HTTPException 500: Se o modelo não estiver carregado.
        HTTPException 400: Se houver erro no processamento dos dados.
    """

    if modelo is None:
        raise HTTPException(status_code=500, detail="O Modelo de IA não foi carregado no servidor.")

    try:
        
        df_input = pd.DataFrame({
            'genres': [jogo.genres],
            'categories': [jogo.categories],
            'publisher': [jogo.publisher], 
            'release_year': [jogo.release_year]
        })


        preco_estimado = modelo.predict(df_input)[0]

        return {
            "status": "sucesso",
            "preco_estimado": round(preco_estimado, 2),
            "moeda": "EUR"
        }

    except Exception as e:
        # Retorna erro 400 (Bad Request) se os dados estiverem num formato que quebra o código
        print(f"Erro na previsão: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    """
    Rota de verificação de saúde da API (Health Check).
    """
    return {"mensagem": "API Steam Predictor Online "}