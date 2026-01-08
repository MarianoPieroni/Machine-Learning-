import streamlit as st
import requests
from joblib import load
import pandas as pd


#streamlit run app.py

# Configuração da Página
st.set_page_config(page_title="Steam Predictor")

# Título e Estilo
st.title("Steam Price Predictor AI")
st.markdown("Bem-vindo! Configure os detalhes do jogo abaixo para prever o preço.")

# Endereço da sua API (A cozinha)
API_URL = "http://127.0.0.1:8000/predict"

# --- BARRA LATERAL (OPÇÕES) ---
st.sidebar.header("Configurações do Jogo")

# 1. Carregar Listas para os Menus
# Colocamos num try para não quebrar o site se o arquivo faltar
try:
    lista_generos = load('generos.joblib')
    lista_publishers = load('publisher.joblib')
except:
    st.error("Erro: Arquivos .joblib de lista não encontrados!")
    lista_generos = ["Action", "Adventure"] # Fallback
    lista_publishers = ["Ubisoft", "Other"]

# 2. Inputs do Usuário (Interface Gráfica)

# Gêneros: Multiselect permite escolher vários!
generos_selecionados = st.sidebar.multiselect(
    "Escolha os Gêneros:",
    options=lista_generos,
    default=lista_generos[0] # Começa com o primeiro selecionado
)

# Publisher: Selectbox permite escolher um
publisher_selecionada = st.sidebar.selectbox(
    "Escolha a Publisher:",
    options=lista_publishers
)

# Ano: Number Input
ano_selecionado = st.sidebar.number_input(
    "Ano de Lançamento:",
    min_value=1990,
    max_value=2030,
    value=2025
)

# --- ÁREA PRINCIPAL (RESULTADO) ---

# Mostra o que o usuário escolheu
st.write("### Resumo do Jogo")
st.write(f"**Publisher:** {publisher_selecionada}")
st.write(f"**Ano:** {ano_selecionado}")
# Junta a lista de generos numa string com ; (ex: "Action;RPG") para a API entender
generos_string = ";".join(generos_selecionados)
st.write(f"**Gêneros:** {generos_string}")

st.markdown("---")

# Botão de Previsão
if st.button("💰 Calcular Preço Sugerido", type="primary"):
    
    # 1. Monta o pacote de dados
    dados_jogo = {
        "genres": generos_string,
        "publisher": publisher_selecionada,
        "release_year": int(ano_selecionado)
    }

    # 2. Barra de progresso (só pra ficar bonito)
    with st.spinner('Consultando a Inteligência Artificial...'):
        try:
            # 3. Manda para a API
            response = requests.post(API_URL, json=dados_jogo)
            
            if response.status_code == 200:
                resultado = response.json()
                preco = resultado['preco_estimado']
                
                # 4. Mostra o resultado GRANDE
                st.success("Previsão realizada com sucesso!")
                st.metric(label="Preço Estimado", value=f"€ {preco:.2f}")
                
            else:
                st.error(f"Erro na API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Erro Crítico: A API não está rodando!")
            st.info("Dica: Verifique se você rodou 'uvicorn api:app' no outro terminal.")