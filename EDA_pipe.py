import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer

def Read_Data():

    df = pd.read_csv("a_steam_data_2021_2025.csv")
    df_clean = df.copy()
    
    print("DATASET ORIGINAL")
    print(f"Dimensoes: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    
    return df, df_clean

def Analyze_Data(df_clean):

    numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df_clean.select_dtypes(include=['object']).columns.tolist()

    variables = {
        'Numericas': numericas,     
        'Categoricas': categoricas, 
        'Texto/Listas': ['genres', 'tags', 'categories'] # Colunas que precisam de tratamento especial
    }
    print("\n--- CLASSIFICAÇÃO DAS VARIÁVEIS ---")
    print(f"Numéricas ({len(numericas)}): {numericas}")
    print(f"Categóricas ({len(categoricas)}): {categoricas}")


    print("\n--- ESTATÍSTICAS DESCRITIVAS (Antes da Limpeza) ---")
    print(df_clean.describe(include='all')) 
    
    print("\n--- VERIFICAÇÃO DE DADOS FALTANTES ---")
    print(df_clean.isnull().sum())

    total_duplicados = df_clean.duplicated().sum()
    print(f"Linhas duplicadas: {total_duplicados}")

    return df_clean


def Clear_Data(df_clean):

    #clear data
    df_clean = df_clean.drop_duplicates() #n temos duplicatas mas vou deixar com
    df_clean = df_clean.dropna(subset=['genres']) #remove os jogos sem generos
    df_clean = df_clean.dropna(subset=['developer', 'publisher','categories']) #remove os jogos sem dev e publisher, poderiamos mudar para Unknown
    df_clean = df_clean.dropna(subset=['price'])
    #tratando outliers pq o randon perdeu // nao foi necessario apenas em limpar a coluna appid ja foi o suficiente pra ganhar // é necessario para melhorar o r2
    df_clean = df_clean[df_clean['price'] < 100]

    if 'release_year' in df_clean.columns:
        df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
        df_clean = df_clean.dropna(subset=['release_year']) # Remove os que deram erro

    #preenche numeros nos numericos restantes
    cols_numericas = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[cols_numericas] = df_clean[cols_numericas].fillna(0)
    
    print(f"Total de linhas após limpeza: {df_clean.shape[0]}")

    print("\nDADOS NULOS APOS LIMPEZA")
    print(df_clean.isnull().sum())
    
    return df_clean

def Split_Data(df_tratado):
    print("\nTreino 80% / Teste 20%")
    
    target = 'price'
        
    X = df_tratado.drop(columns=[target]) # Perguntas (Dados)
    y = df_tratado[target]                # Resposta (Preço)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Treino: {X_train.shape[0]} jogos")
    print(f"Teste:  {X_test.shape[0]} jogos")
    
    return X_train, X_test, y_train, y_test

def Train_Models(X_train, X_test, y_train, y_test):
    print("\nTREINO E COMPARAÇÃO")
    
    # Regressão Linear, O mínimo aceitável
    print("\nTreinando Regressão Linear (baseline)")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Previsão
    pred_lr = model_lr.predict(X_test)
    
    # Avaliação (Erro Médio)
    maen_lr = mean_absolute_error(y_test, pred_lr)
    r2_lr = r2_score(y_test, pred_lr)
    print(f"Erro Médio (Regressão Linear): {maen_lr:.2f} euros")
    print(f"Score R²: {r2_lr:.4f}") #medir a qualdiade, quanto mais perrto do 1 melhor
    
    # Random Forest
    print("\nTreinando Random Forest")
    # n_estimators=100 significa que ele cria 100 árvores mentais
    model_rf = RandomForestRegressor(   #melhorando o random, antes so dava o tamanho da arvore, aplicamos o hiper
        n_estimators=200,      # 200 arvores
        max_depth=10,          # Limita a profundidade (evitar o overfitting)
        min_samples_leaf=2,    # Garante que cada folha tenha pelo menos 2 dados
        random_state=42,
        n_jobs=-1              # Acelera o treino
    )
    model_rf.fit(X_train, y_train)
    
    # Previsão
    pred_rf = model_rf.predict(X_test)
    
    # Avaliação
    maen_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)
    print(f"Erro Médio (Random Forest): {maen_rf:.2f} euros")
    print(f"Score R²: {r2_rf:.4f}")
    
    print("\nRESULTADO FINAL")
    if maen_rf < maen_lr:
        melhoria = maen_lr - maen_rf
        print(f"O Random Forest venceu")
        print(f"Ele erra {melhoria:.2f} euros a menos")
        return model_rf
    else:
        print("O Randon Florest perdeu")
        return model_lr


def to_list(x):
    """
    Converte o input para lista de forma segura.
    Funciona para DataFrame com 1 linha ou várias.
    """
    if isinstance(x, pd.DataFrame):
        # Pega a primeira coluna (independente do nome) e converte para lista
        return x.iloc[:, 0].tolist()
    
    # Se já for uma Series ou array
    return x.tolist()

def split_semicolon(text):
    """
    CORREÇÃO 1: Divide, remove espaços e converte para MINÚSCULO.
    Isso garante que 'Action; RPG' vire ['action', 'rpg']
    """
    if not isinstance(text, str):
        return []
    return [t.strip().lower() for t in text.split(';')]

def limpar_publisher(x):
    """
    CORREÇÃO 2: Função nova para limpar a Publisher antes do OneHotEncoder.
    Converte tudo para minúsculo e remove espaços.
    """
    # Se for DataFrame, pega a série de texto
    if isinstance(x, pd.DataFrame):
        text_series = x.iloc[:, 0].astype(str)
    else:
        text_series = x.astype(str)
    
    return text_series.str.lower().str.strip().to_frame()

def pipeline(X_train, X_test, y_train, y_test):

    print("\n[PIPELINE] Configurando e Treinando Pipeline Blindado...")
    
    # 1. Configurar transformadores
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

    genres_transformer = Pipeline(steps=[
        ('formatter', FunctionTransformer(to_list)), 
        ('vect', CountVectorizer(tokenizer=split_semicolon, ngram_range=(1, 2), max_features=200)) 
    ])
    
    publisher_transformer = Pipeline(steps=[
        ('limpeza', FunctionTransformer(limpar_publisher)), 
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=1000))
    ])

    # 2. Juntar tudo
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['release_year']),
            ('cat_genres', genres_transformer, ['genres']),
            ('cat_pub', publisher_transformer, ['publisher'])
        ],
        remainder='drop'
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3, # CONTROLA overfitting
        random_state=42,
        n_jobs=-1)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # 4. Treinar
    pipeline.fit(X_train, y_train)
    
    # 5. Avaliar
    pred_rf = pipeline.predict(X_test)
    maen_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)
    print(f"Erro Médio (Random Forest): {maen_rf:.2f} euros")
    print(f"Score R²: {r2_rf:.4f}")

    # --- CORREÇÃO DA EXTRAÇÃO ---
    lista_generos_aprendidos = []
    lista_publishers = []
    
    try:
        # Extrair Gêneros
        vect_step = pipeline.named_steps['preprocessor'].named_transformers_['cat_genres'].named_steps['vect']
        lista_generos_aprendidos = vect_step.get_feature_names_out()

        # Extrair Publishers (CORRIGIDO)
        # 1. Acessamos a Pipeline de Publisher ('cat_pub')
        pub_pipe = pipeline.named_steps['preprocessor'].named_transformers_['cat_pub']
        # 2. Dentro dela, acessamos o passo 'encoder'
        enc_step = pub_pipe.named_steps['encoder']
        
        raw_pub_names = enc_step.get_feature_names_out()
        
        # O prefixo muda para 'x0_' por causa da função de limpeza, então removemos 'x0_'
        lista_publishers = [name.replace('publisher_', '').replace('x0_', '') for name in raw_pub_names]

    except Exception as e:
        print(f"Aviso: Não foi possível extrair listas ({e})")
        lista_generos_aprendidos = []
        lista_publishers = []
    
    return pipeline, lista_generos_aprendidos, lista_publishers

def main():
    df, df_clean = Read_Data()

    if df is not None:
        #analize
        df_clean = Analyze_Data(df_clean)
        #limpeza
        df_clean = Clear_Data(df_clean)

        #pipe
        cols_to_use = ['genres', 'publisher', 'release_year', 'price']
        df_pronto_para_split = df_clean[cols_to_use]
        X_train, X_test, y_train, y_test = Split_Data(df_pronto_para_split)

        #treino
        if X_train is not None:
            melhor_modelo,lista_generos,lista_publishers = pipeline(X_train, X_test, y_train, y_test)
            #criar o joblib
            if melhor_modelo is not None:
                from joblib import dump
                dump(melhor_modelo, 'steam_price_model.joblib')
                dump(lista_generos, 'generos.joblib')
                dump(lista_publishers, 'publisher.joblib')
                print("joblib criado")

    return df_clean

""" if __name__ == "__main__":
    main() """