import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

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
    df_clean = df_clean.dropna(subset=['developer', 'publisher']) #remove os jogos sem dev e publisher, poderiamos mudar para Unknown
    df_clean = df_clean.dropna(subset=['price'])
    #preenche numeros nos numericos restantes
    cols_numericas = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[cols_numericas] = df_clean[cols_numericas].fillna(0)
    
    print(f"Total de linhas após limpeza: {df_clean.shape[0]}")

    return df_clean

def main():
    df, df_clean = Read_Data()

    if df is not None:
        df_clean = Analyze_Data(df_clean)
        df_clean = Clear_Data(df_clean)

    return df_clean

if __name__ == "__main__":
    df_clean = main()