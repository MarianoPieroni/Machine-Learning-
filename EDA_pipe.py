import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer

def Read_Data():
    """
    Lê o dataset original e retorna uma cópia limpa para processamento.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados brutos do CSV.
    """

    df = pd.read_csv("a_steam_data_2021_2025.csv")
    df_clean = df.copy()
    
    print("DATASET ORIGINAL")
    print(f"Dimensoes: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    
    return df_clean

def Analyze_Data(df_clean):
    """
    Realiza uma análise exploratória inicial para identificar tipos de variáveis e dados faltantes.

    Args:
        df_clean (pd.DataFrame): DataFrame a ser analisado.
    
    Returns:
        pd.DataFrame: O mesmo DataFrame, inalterado.
    """
    numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df_clean.select_dtypes(include=['object']).columns.tolist()
    
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
    """
    Executa a limpeza dos dados, removendo nulos críticos, duplicatas e outliers.

    Args:
        df_clean (pd.DataFrame): DataFrame bruto.

    Returns:
        pd.DataFrame: DataFrame limpo e pronto para o split.
    """

    # Remove o que nao for numero da coluna
    if 'release_year' in df_clean.columns:
        df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
        df_clean = df_clean.dropna(subset=['release_year']) 

    #clear data com menos linhas
    #remove os faltantes de cada variavel e remove as duplicatas
    cols_check = ['genres', 'developer', 'publisher', 'categories', 'price', 'release_year']
    df_clean = df_clean.dropna(subset=cols_check).drop_duplicates()

    #tratando outliers pq o randon perdeu // nao foi necessario apenas em limpar a coluna appid ja foi o suficiente pra ganhar // é necessario para melhorar o r2
    df_clean = df_clean[df_clean['price'] < 100]
    df_clean = df_clean[df_clean['price'] > 0.5]

    #tirei o codigo que complementa com 0 os valores nulos, esta redundante ja que tiramos todos os nulos antes
    
    print(f"Total de linhas após limpeza: {df_clean.shape[0]}")

    print("\nDADOS NULOS APOS LIMPEZA")
    print(df_clean.isnull().sum())
    
    return df_clean

def Split_Data(df_clean):
    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        df_clean (pd.DataFrame): DataFrame limpo.

    Returns:
        tuple: X_train, X_test, y_train, y_test prontos para o pipeline.
    """

    print("\nTreino 80% / Teste 20%")

    cols_to_use = ['genres', 'categories', 'publisher', 'release_year', 'price']
    X = df_clean[cols_to_use].drop(columns=['price'])
    y = df_clean['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    
    return X_train, X_test, y_train, y_test


#FUNÇÕES AUXILIARES DO PIPELINE 
def to_list(x):
    """
    Garante que a entrada para o vetorizador seja uma lista.
    Essencial para compatibilidade com DataFrames de coluna única.
    """

    #Converte o input para lista de forma segura.
    if isinstance(x, pd.DataFrame):
        # Pega a primeira coluna (independente do nome) e converte para lista
        return x.iloc[:, 0].tolist()
    
    return x.tolist()

def split_semicolon(text):
    """
    Tokenizador personalizado para separar strings baseadas em ponto e vírgula.
    Ex: 'Action; RPG' -> ['action', 'rpg']
    """

    # Divide, remove espaços e converte para MINÚSCULO ('Action; RPG' -> ['action', 'rpg'])
    if not isinstance(text, str):
        return []
    return [t.strip().lower() for t in text.split(';')]

def limpar_publisher(x):
    """
    Padroniza os nomes dos publishers para evitar duplicidade no OneHotEncoder.
    Remove espaços e converte para minúsculas.
    """

    #limpar/padronizar a Publisher antes do OneHotEncoder.
    #Converte tudo para minúsculo e remove espaços.
    if isinstance(x, pd.DataFrame):
        text_series = x.iloc[:, 0].astype(str)
    else:
        text_series = x.astype(str)
    
    return text_series.str.lower().str.strip().to_frame()

def Treinar_pipeline(X_train, X_test, y_train, y_test):
    """
    Configura e treina o Pipeline de Machine Learning.
    
    O Pipeline inclui:
    1. Imputação de mediana para anos.
    2. CountVectorizer para listas de texto (Gêneros/Categorias).
    3. OneHotEncoder para Publishers (após limpeza).
    4. Modelo RandomForestRegressor.
    """

    print("\nTreinando Pipeline")

    # transformadores
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

    text_list_transformer = Pipeline(steps=[
        ('formatter', FunctionTransformer(to_list)), 
        ('vect', CountVectorizer(tokenizer=split_semicolon, ngram_range=(1, 2), max_features=200)) 
    ])
    
    publisher_transformer = Pipeline(steps=[
        ('limpeza', FunctionTransformer(limpar_publisher)), 
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=1000))
    ])

    # Juntar tudo
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['release_year']),
            ('cat_genres', text_list_transformer, ['genres']),
            ('cat_categories', text_list_transformer, ['categories']),
            ('cat_pub', publisher_transformer, ['publisher'])
        ],
        remainder='drop'
    )

    model_rf = Comparar_modelos(X_train, X_test, y_train, y_test,preprocessor)
    
    return model_rf

def Comparar_modelos(X_train, X_test, y_train, y_test,preprocessor):

    """ model_lr = LinearRegression()
    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_lr)
    ])
    
    # Treinar
    pipeline_lr.fit(X_train, y_train)
    
    # Avaliar modelo
    pred_lr = pipeline_lr.predict(X_test)
    maen_lr = mean_absolute_error(y_test, pred_lr)
    r2_lr = r2_score(y_test, pred_lr) """



    """ model_gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,       # Geralmente usa arvores menores que o Random Forest
        min_samples_leaf=2,
        random_state=42
    )
    pipeline_gb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_gb)
    ])
    
    # Treinar
    pipeline_gb.fit(X_train, y_train)
    
    # Avaliar modelo
    pred_gb = pipeline_gb.predict(X_test)
    maen_gb = mean_absolute_error(y_test, pred_gb)
    r2_gb = r2_score(y_test, pred_gb) """



    model_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=4, # CONTROLA overfitting
        random_state=42,
        n_jobs=-1)

    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_rf)
    ])
    
    # Treinar
    pipeline_rf.fit(X_train, y_train)
    
    # Avaliar modelo
    pred_rf = pipeline_rf.predict(X_test)
    maen_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)



    """ print("\nComparação dos modelos")

    print("\nRegressão Linear")
    print(f"Erro Médio: {maen_lr:.2f} euros")
    print(f"Score R²: {r2_lr:.4f}")

    print("\nGradient Boosting")
    print(f"Erro Médio: {maen_gb:.2f} euros")
    print(f"Score R²: {r2_gb:.4f}") """

    print("\nRandom Forest")
    print(f"Erro Médio: {maen_rf:.2f} euros")
    print(f"Score R²: {r2_rf:.4f}")

    return pipeline_rf



def Salvar_variaveis(pipeline):
    """
    Salva o modelo treinado e os artefatos (listas de categorias) em arquivos .joblib.
    Remove o prefixo 'publisher_' gerado pelo OneHotEncoder para limpeza visual.
    """

    # Acesso aos transformadores dentro do pipeline
    pre = pipeline.named_steps['preprocessor']
        
    vocab_genres = pre.named_transformers_['cat_genres']['vect'].get_feature_names_out()
    vocab_cats = pre.named_transformers_['cat_categories']['vect'].get_feature_names_out()
        
    # Publishers (tratamento para OHE)
    raw_pubs = pre.named_transformers_['cat_pub']['encoder'].get_feature_names_out()
    vocab_pubs = [name.replace('publisher_', '') for name in raw_pubs]
        
    dump(vocab_genres, 'generos.joblib')
    dump(vocab_cats, 'categorias.joblib')        
    dump(vocab_pubs, 'publisher.joblib')
    dump(pipeline, 'steam_price_model.joblib')
    print("joblibs criados")


def main():
    
    df = Read_Data()
    Analyze_Data(df) 
    df_clean = Clear_Data(df)
    X_train, X_test, y_train, y_test = Split_Data(df_clean)
    modelo_final = Treinar_pipeline(X_train, X_test, y_train, y_test)
    Salvar_variaveis(modelo_final)

        #Regressão Linear: Diria: "Bom, parece que sobe 5 euros por ano. Então em 2030 será €75." (Ela traça uma linha infinita).
        #O ano 2030 é maior que 2025? Sim. Eu tenho dados depois de 2025? Não. Então a melhor
        #resposta que tenho é a média do último grupo que conheço (2025)." Resultado: €50.
    
 #   Treinar_pipeline(X_train, X_test, y_train, y_test)
""" if __name__ == "__main__":
    main() """