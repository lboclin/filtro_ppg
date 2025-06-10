# -*- coding: utf-8 -*-
"""
Script de Análise de Resultados de BPM.

Este script lê todos os arquivos .csv de uma pasta de resultados,
calcula as estatísticas de média e mediana para a coluna 'bpm' de cada arquivo,
e imprime um resumo formatado no terminal.

"""

import os
import pandas as pd
import numpy as np

# --- 1. Configurações ---
# Define o diretório onde os arquivos .csv de resultados estão localizados.
# O script espera que esta pasta esteja dentro da pasta raiz do projeto.
INPUT_DIR = "results/"

# --- 2. Execução Principal ---
if __name__ == "__main__":
    
    # Validação do diretório de entrada para garantir que o caminho existe.
    if not os.path.isdir(INPUT_DIR):
        print(f"🚨 ERRO: Diretório de entrada não encontrado: '{os.path.abspath(INPUT_DIR)}'")
        # Fornece instruções claras se a pasta não for encontrada.
        print("\nPor favor, siga estes passos:")
        print(f"1. Crie a pasta '{INPUT_DIR}' no seu projeto.")
        print("2. Mova os arquivos .csv gerados pelo último script de cálculo de BPM para dentro dela.")
        print("3. Execute este script de análise novamente.")
        exit()

    # Cria uma lista com o nome de todos os arquivos que terminam com .csv no diretório.
    # A lista é ordenada para garantir uma ordem de processamento consistente.
    csv_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])
    
    # Verifica se algum arquivo foi encontrado antes de prosseguir.
    if not csv_files:
        print(f"Nenhum arquivo .csv encontrado em '{INPUT_DIR}'.")
        exit()

    print(f"--- Análise dos Resultados Finais de BPM ({len(csv_files)} arquivos) ---")

    # Itera sobre cada nome de arquivo encontrado.
    for filename in csv_files:
        file_path = os.path.join(INPUT_DIR, filename)
        
        try:
            # Extrai o tipo de atividade do nome do arquivo.
            # Assume um padrão de nome como "s1_walk_...csv" e pega o segundo elemento ('walk').
            activity = filename.split('_')[1].capitalize()
            
            # Carrega o arquivo CSV para um DataFrame do pandas.
            df = pd.read_csv(file_path)

            # Validação dos dados: verifica se o DataFrame está vazio ou se a coluna 'bpm' não tem dados válidos.
            if df.empty or 'bpm' not in df.columns or df['bpm'].isnull().all():
                print(f"{activity} ({filename}): Dados insuficientes ou vazios.")
                continue # Pula para o próximo arquivo.

            # Calcula a média e a mediana da coluna 'bpm'.
            mean_bpm = df['bpm'].mean()
            median_bpm = df['bpm'].median()

            # Imprime os resultados no formato solicitado.
            # "median_bpm:.2f" formata o número para ter duas casas decimais.
            print(f"{activity}: mediana->{median_bpm:.2f}, media->{mean_bpm:.2f}")

        except IndexError:
            # Captura erros caso o nome do arquivo não siga o padrão esperado (ex: sem '_').
            print(f"AVISO: Não foi possível determinar a atividade para o arquivo '{filename}'. Pulando.")
        except Exception as e:
            # Captura qualquer outro erro que possa ocorrer durante o processamento do arquivo.
            print(f"ERRO ao processar o arquivo {filename}: {e}")

    print("\n--- Análise Concluída ---")