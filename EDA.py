import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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

    #preenche numeros nos numericos restantes
    cols_numericas = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[cols_numericas] = df_clean[cols_numericas].fillna(0)
    
    print(f"Total de linhas após limpeza: {df_clean.shape[0]}")

    print("\nDADOS NULOS APOS LIMPEZA")
    print(df_clean.isnull().sum())
    
    return df_clean

def Transform_Data(df_clean):
    print("\nTRATAR DADOS (Transformar texto em número)")
    top_genres = []
    # //
    if 'genres' in df_clean.columns:
        # 15 generos mais comuns

        df_clean['genres'] = df_clean['genres'].astype(str).str.strip()
        all_genres = df_clean['genres'].str.split(';').explode()
        all_genres = all_genres.str.strip()
        top_genres = all_genres.value_counts().head(15).index.tolist()
        
     #   print(f"{top_genres}")
        
        for genre in top_genres:
            # Cria coluna binario
            col_name = f'Gen_{genre.strip()}'
            df_clean[col_name] = df_clean['genres'].apply(lambda x: 1 if genre in str(x) else 0)
    
    # //
    if 'release_year' in df_clean.columns:
        df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
        # Preenche anos vazios com a média ou mediana (ex: 2015) para não perder linhas
        mediana_ano = df_clean['release_year'].median()
        df_clean['release_year'] = df_clean['release_year'].fillna(mediana_ano)
        print(f"Variável 'release_year' incluída no modelo.")
    # //


    # Filtramos para ficar apenas com binario
    df_tratado = df_clean.select_dtypes(include=[np.number])

    #retiramos a coluna do appid para o randon ganhar
    if 'appid' in df_tratado.columns:
        df_tratado = df_tratado.drop(columns=['appid']) 

    return df_tratado, top_genres

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
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
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


def main():
    df, df_clean = Read_Data()

    if df is not None:
        #analize
        df_clean = Analyze_Data(df_clean)
        #limpeza
        df_clean = Clear_Data(df_clean)
        #tratamento binario
        df_clean,lista_generos = Transform_Data(df_clean)
        #divisao de treino e teste
        X_train, X_test, y_train, y_test = Split_Data(df_clean)
        #treino
        if X_train is not None:
            melhor_modelo = Train_Models(X_train, X_test, y_train, y_test)
            #criar o joblib
            if melhor_modelo is not None:
                from joblib import dump
                dump(melhor_modelo, 'steam_price_model.joblib')
                dump(lista_generos, 'generos.joblib')
                print("joblib criado")
    #notas
    #ao analizar o r2 vimos que conseguimos prever apenas 5% do preço a partir do generos
    #com isso concluimos que genero nao define o preço
    #se tratarmos os outliers o r2 sobe para 12% em media 
    #se nao tiramos o id o randon florest tem dificuldades para prever fazendo o rl ganhar
    #achei fraco prever apenas com o preço e fiz em relaçao ao ano e nao mudou nada o R2
    #com isso vemos que a base de dados é "limitada"
    #tentei adicionar o categories para prever junto com o ano e o genero e o r2 ficou negativo, entao tirei

    return df_clean