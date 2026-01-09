import pandas as pd
import sys
import os
from joblib import load
import EDA_pipe  # Importa o seu EDA.py

def menu():
    print("STEAM PICES PREDICTOR")
    print("1. Treinar IA (OBS: SE ja tiver o joblib não precisa ser feita)")
    print("2. Fazer Previsão")
    print("3. Sair")
    return input("Escolha: ")

def modo_previsao():
    print("\nPREVISÃO")
    arquivo = 'steam_price_model.joblib'
    
    if not os.path.exists(arquivo):
        print("Use a Opção 1 primeiro para criar o joblib")
        return

    #carrega o modelo (joblib)
    modelo = load(arquivo)
    lista_generos = load('generos.joblib')
    lista_publisher = load('publisher.joblib')
    lista_categorias = load('categorias.joblib')

    print("\nGêneros disponiveis")
    print(", ".join(lista_generos))
    entrada_generos = input("\nGêneros: ")

    print("\Categorias disponiveis")
    print(", ".join(lista_categorias))
    entrada_categorias = input("Categorias: ")

    print("\nPublishers disponiveis")
    print(", ".join(lista_publisher))
    entrada_publisher = input("\publisher: ")    

    entrada_ano = int(input("Ano de Lançamento: "))

    dados_input = pd.DataFrame({
        'release_year': [entrada_ano],
        'genres': [entrada_generos],
        'categories': [entrada_categorias],
        'publisher': [entrada_publisher]
    })

    X_novo = pd.DataFrame(dados_input)
    preco = modelo.predict(X_novo)[0]
    print(f" PREÇO ESTIMADO: {preco:.2f} euros")

"""     validar_genero = 0
    validar_publisher = 0 """
    
    #transforma o genero digitado em binario 
"""     for genero in lista_generos:
        col_tecnica = f"Gen_{genero}"
        if genero.lower() in entrada_generos.lower():
            dados_input[col_tecnica] = [1]
            validar_genero +=1
        else:
            dados_input[col_tecnica] = [0]
    


    if validar_genero == 0:
        print("Nome do genero errado.")
        return

    try:
        X_novo = pd.DataFrame(dados_input)      #converte os dados para a Scikit-Learn
        preco = modelo.predict(X_novo)[0]       #parte principal do codigo de prever dados, faz os calculos aprendidos
        print(f"\nPREÇO ESTIMADO: {preco:.2f} euros")
    except Exception as e:
        print(f"\nErro na previsão: {e}") """



def main():
    while True:
        op = menu()
        if op == '1':
            EDA_pipe.main()
        elif op == '2':
            modo_previsao()
        elif op == '3':
            sys.exit()
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()