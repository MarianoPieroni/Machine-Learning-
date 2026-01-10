import streamlit as st
import requests
from joblib import load

# streamlit run api_visual.py

# Configuração da Página
st.set_page_config(page_title="Steam Predictor", layout="centered")

# Título
st.title("Steam Price Predictor AI")
st.markdown("Bem-vindo! Configure os detalhes do jogo abaixo para prever o preço.")

# Endereço da API
API_URL = "http://127.0.0.1:8000/predict"

# --- BARRA LATERAL (OPÇÕES) ---
st.sidebar.header("Configurações do Jogo")

#Carregar Listas para os Menus
try:
    # Carregamos os dados brutos
    raw_generos = load('generos.joblib')
    raw_publishers = load('publisher.joblib')
    raw_categorias = load('categorias.joblib')

    # Verifica se é um array do Numpy (que tem a função .tolist())
    # Se for, converte para lista normal do Python. Se não, usa como está.
    lista_generos = raw_generos.tolist() if hasattr(raw_generos, 'tolist') else raw_generos
    lista_publishers = raw_publishers.tolist() if hasattr(raw_publishers, 'tolist') else raw_publishers
    lista_categorias = raw_categorias.tolist() if hasattr(raw_categorias, 'tolist') else raw_categorias

except Exception as e:
    st.error(f"Aviso: Não foi possível carregar as listas ({e}). Usando padrões.")
    # Fallback para não travar o site
    lista_generos = ["Action", "Adventure", "Indie"] 
    lista_publishers = ["Ubisoft", "Other"]
    lista_categorias = ["Single-player", "Multi-player"]

#Inputs

# Gêneros
generos_selecionados = st.sidebar.multiselect(
    "Escolha os Gêneros:",
    options=lista_generos,
    # Verifica o tamanho da lista (len) em vez de usar o array direto
    default=lista_generos[0] if len(lista_generos) > 0 else None
)

# Categorias
categorias_selecionadas = st.sidebar.multiselect(
    "Categorias:",
    options=lista_categorias,
    default=lista_categorias[0] if len(lista_categorias) > 0 else None
)

# Publisher
publisher_selecionada = st.sidebar.selectbox(
    "Escolha a Publisher:",
    options=lista_publishers
)

# Ano
ano_selecionado = st.sidebar.number_input(
    "Ano de Lançamento:",
    min_value=2021,
    max_value=2025,
    value=2025
)

# --- ÁREA PRINCIPAL (RESULTADO) ---

st.write("### Resumo do Jogo")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Publisher:** {publisher_selecionada}")
    st.write(f"**Ano:** {ano_selecionado}")

with col2:
    # Junta a lista numa string com ; (ex: "Action;RPG") para a API entender
    generos_string = ";".join(generos_selecionados)
    categorias_string = ";".join(categorias_selecionadas)
    
    st.write(f"**Gêneros:** {generos_string}")
    st.write(f"**Categorias:** {categorias_string}")

st.markdown("---")

# Botão de Previsão
if st.button("Calcular Preço Sugerido", type="primary"):
    
    # 1. Monta o pacote de dados
    dados_jogo = {
        "genres": generos_string,
        "categories": categorias_string,
        "publisher": publisher_selecionada,
        "release_year": int(ano_selecionado)
    }

    # 2. Barra de progresso
    with st.spinner('Consultando a Inteligência Artificial...'):
        try:
            # 3. Manda para a API
            response = requests.post(API_URL, json=dados_jogo)
            
            if response.status_code == 200:
                resultado = response.json()
                preco = resultado['preco_estimado']
                
                # 4. Mostra o resultado
                st.success("Previsão realizada com sucesso!")
                st.metric(label="Preço Estimado", value=f"€ {preco:.2f}")
                
            else:
                st.error(f"Erro na API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Erro Crítico: A API não está rodando!")
            st.info("Dica: Verifique se você rodou 'uvicorn api:app' no outro terminal.")