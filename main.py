import EDA # Importa o seu arquivo
import sys

def menu():
    print("STEAM PREDICTOR")
    print("1. Treinar IA (Rodar EDA)")
    print("2. Fazer Previsão")
    print("3. Sair")
    return input("Escolha: ")

def main():
    while True:
        escolha = menu()
        
        if escolha == '1':
            EDA.main()
            
        #elif escolha == '2':

        elif escolha == '3':
            sys.exit()
        else:
            print("Opção inválida")
            
if __name__ == "__main__":
    main()