import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split

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
    #preenche numeros nos numericos restantes
    cols_numericas = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[cols_numericas] = df_clean[cols_numericas].fillna(0)
    
    print(f"Total de linhas após limpeza: {df_clean.shape[0]}")

    print("\n--- DADOS NULOS APOS LIMPEZA ---")
    print(df_clean.isnull().sum())
    
    return df_clean

def Transform_Data(df_clean):
    print("\n--- TRATAR DADOS (Transformar texto em número) ---")
    if 'genres' in df_clean.columns:
        # Pega os 10 gêneros mais comuns
        top_genres = pd.Series(', '.join(df_clean['genres']).split(', ')).value_counts().head(10).index
        
        print(f"Criando colunas para: {top_genres.tolist()}")
        
        for genre in top_genres:
            # Cria coluna binária (1 se tiver o gênero, 0 se não)
            col_name = f'Gen_{genre.strip()}'
            df_clean[col_name] = df_clean['genres'].apply(lambda x: 1 if genre in str(x) else 0)
            
    # Filtramos para ficar apenas com números (Preço, Notas, e os novos Gêneros 0/1)
    df_tratado = df_clean.select_dtypes(include=[np.number])

    return df_tratado

def Split_Data(df_tratado):
    print("\n--- DIVISÃO (Treino 80% / Teste 20%) ---")
    
    target = 'price'
        
    X = df_tratado.drop(columns=[target]) # Perguntas (Dados)
    y = df_tratado[target]                # Resposta (Preço)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Treino: {X_train.shape[0]} jogos")
    print(f"Teste:  {X_test.shape[0]} jogos")
    
    return X_train, X_test, y_train, y_test


def main():
    df, df_clean = Read_Data()

    if df is not None:
        #analize
        df_clean = Analyze_Data(df_clean)
        #limpeza
        df_clean = Clear_Data(df_clean)
        #tratamento binario
        df_clean = Transform_Data(df_clean)
        #divisao de treino e teste
        X_train, X_test, y_train, y_test = Split_Data(df_clean)

    return df_clean

if __name__ == "__main__":
    df_clean = main()