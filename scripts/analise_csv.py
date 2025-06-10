import os
import pandas as pd
import numpy as np

# --- Configurações ---
# O script irá procurar por esta pasta dentro de "results/"
INPUT_DIR = "results/ground_truth/"

if __name__ == "__main__":
    # Validação do diretório de entrada
    if not os.path.isdir(INPUT_DIR):
        print(f"🚨 ERRO: Diretório de entrada não encontrado: '{os.path.abspath(INPUT_DIR)}'")
        print("\nPor favor, siga estes passos:")
        print(f"1. Crie a pasta '{INPUT_DIR}' no seu projeto.")
        print("2. Mova os arquivos .csv gerados pelo último script de cálculo de BPM para dentro dela.")
        print("3. Execute este script de análise novamente.")
        exit()

    # Encontra todos os arquivos .csv no diretório
    csv_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])
    
    if not csv_files:
        print(f"Nenhum arquivo .csv encontrado em '{INPUT_DIR}'.")
        exit()

    print(f"--- Análise dos Resultados Finais de BPM ({len(csv_files)} arquivos) ---")

    for filename in csv_files:
        file_path = os.path.join(INPUT_DIR, filename)
        
        try:
            # Extrai o tipo de atividade do nome do arquivo (ex: 's1_walk_...' -> 'walk')
            activity = filename.split('_')[1].capitalize()
            
            # Carrega o arquivo CSV
            df = pd.read_csv(file_path)

            # Verifica se o dataframe está vazio ou não tem dados válidos de bpm
            if df.empty or 'bpm' not in df.columns or df['bpm'].isnull().all():
                print(f"{activity} ({filename}): Dados insuficientes ou vazios.")
                continue

            # Calcula a média e a mediana
            mean_bpm = df['bpm'].mean()
            median_bpm = df['bpm'].median()

            # Imprime no formato solicitado
            print(f"{activity}: mediana->{median_bpm:.2f}, media->{mean_bpm:.2f}")

        except IndexError:
            # Caso o nome do arquivo não siga o padrão esperado
            print(f"AVISO: Não foi possível determinar a atividade para o arquivo '{filename}'. Pulando.")
        except Exception as e:
            print(f"ERRO ao processar o arquivo {filename}: {e}")

    print("\n--- Análise Concluída ---")