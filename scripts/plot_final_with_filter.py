# -*- coding: utf-8 -*-
"""
Script de Pós-processamento e Visualização Final de Resultados.

Este script carrega os resultados de BPM calculados (tanto do algoritmo PPG
quanto do Ground Truth de ECG), aplica um filtro de outliers para suavizar
as séries temporais, e gera um gráfico comparativo sobreposto para cada
registro, permitindo uma avaliação visual final da performance do algoritmo.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configurações ---

# Diretórios de entrada com os resultados em CSV.
# Altere os nomes das pastas se estiverem diferentes no seu projeto.
PPG_RESULTS_DIR = "results/bpm_vfinal/" 
ECG_TRUTH_DIR = "results/ground_truth/"

# Diretório de saída para os gráficos de comparação final.
PLOT_OUTPUT_DIR = "outputs/final_comparison_filtered/"

# --- Parâmetros do Filtro de Outliers ---
# Se um valor de BPM estiver a uma distância maior que este valor da mediana geral,
# ele será considerado um outlier e substituído pela própria mediana.
OUTLIER_THRESHOLD_BPM = 15.0

# --- 2. Função Auxiliar para Filtragem ---

def filter_outliers(bpm_series, threshold):
    """
    Substitui outliers em uma série de BPM pela mediana da série.

    Args:
        bpm_series (pd.Series): A série de dados de BPM a ser filtrada.
        threshold (float): A distância máxima da mediana para um ponto não ser considerado outlier.

    Returns:
        pd.Series: A nova série de BPM com os outliers suavizados.
    """
    # Retorna a série original se ela estiver vazia.
    if bpm_series.empty or bpm_series.isnull().all():
        return bpm_series
    
    # Calcula a mediana de toda a série temporal de BPM.
    median_bpm = bpm_series.median()
    
    # Cria uma cópia para evitar modificar os dados originais.
    filtered_series = bpm_series.copy()
    
    # Cria uma máscara booleana que é 'True' para todos os pontos que são outliers.
    outlier_mask = np.abs(filtered_series - median_bpm) > threshold
    
    # Usa a máscara para substituir o valor dos outliers pelo valor da mediana.
    filtered_series[outlier_mask] = median_bpm
    
    return filtered_series

# --- 3. Execução Principal ---
if __name__ == "__main__":
    # Validação e criação de diretórios de saída.
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Diretório de saída para os gráficos: {os.path.abspath(PLOT_OUTPUT_DIR)}")

    # Encontra os arquivos CSV do nosso algoritmo para guiar o processo.
    # Ajuste o sufixo '.csv' se seus arquivos tiverem um nome mais específico.
    ppg_result_files = sorted([f for f in os.listdir(PPG_RESULTS_DIR) if f.endswith(".csv")])
    
    if not ppg_result_files:
        print(f"Nenhum arquivo .csv encontrado em '{PPG_RESULTS_DIR}'.")
        exit()

    print(f"\n--- Gerando Gráficos Finais com Filtro de Outliers ({len(ppg_result_files)} arquivos) ---")

    # Itera sobre cada arquivo de resultado encontrado.
    for ppg_filename in ppg_result_files:
        # Extrai o nome base para encontrar o arquivo de ground truth correspondente.
        base_name = ppg_filename.split('_bpm_results')[0]
        print(f"\n--- Processando e Plotando: {base_name} ---")
        
        try:
            # --- Carregamento e Filtragem dos Dados ---
            
            # Carrega e filtra os dados do nosso algoritmo PPG.
            ppg_path = os.path.join(PPG_RESULTS_DIR, ppg_filename)
            df_ppg = pd.read_csv(ppg_path)
            if not df_ppg.empty:
                df_ppg['bpm_filtered'] = filter_outliers(df_ppg['bpm'], OUTLIER_THRESHOLD_BPM)

            # Encontra, carrega e filtra os dados do Ground Truth (ECG).
            ecg_filename = f"{base_name}_ecg_bpm.csv"
            ecg_path = os.path.join(ECG_TRUTH_DIR, ecg_filename)
            
            df_ecg = None
            if os.path.exists(ecg_path):
                df_ecg = pd.read_csv(ecg_path)
                if not df_ecg.empty:
                    df_ecg['bpm_filtered'] = filter_outliers(df_ecg['bpm_ecg'], OUTLIER_THRESHOLD_BPM)
            else:
                print(f"  ℹ️  Arquivo Ground Truth '{ecg_filename}' não encontrado.")

            # --- Plotagem Sobreposta ---
            plt.figure(figsize=(20, 7))
            
            # Plota a linha do ECG (Ground Truth) como referência principal.
            if df_ecg is not None and not df_ecg.empty:
                plt.plot(df_ecg['tempo_s'], df_ecg['bpm_filtered'], 
                         color='black', linestyle='-', linewidth=2.5, 
                         label='BPM Real (ECG - Pós-Filtro)')

            # Plota a linha do PPG (Nosso Algoritmo) de forma pontilhada para comparação.
            if not df_ppg.empty:
                plt.plot(df_ppg['tempo_s'], df_ppg['bpm_filtered'], 
                         color='deepskyblue', linestyle='--', linewidth=2.0, 
                         label='BPM Estimado (PPG - Pós-Filtro)')
            
            # Configuração de títulos e legendas para clareza.
            activity = base_name.split('_')[1]
            plt.title(f'Comparação Final (Pós-Filtro de Outliers)\nAtividade: {activity.capitalize()} - Registro: {base_name}', fontsize=16)
            plt.xlabel('Tempo da Atividade (s)', fontsize=12)
            plt.ylabel('Batimentos por Minuto (BPM)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.ylim(40, 220) # Eixo Y fixo para facilitar a comparação visual entre gráficos.

            # Salva a figura no diretório de saída.
            output_path = os.path.join(PLOT_OUTPUT_DIR, f"{base_name}_filtered_comparison.png")
            plt.savefig(output_path)
            plt.close() # Libera a memória da figura.
            
            print(f"  ✅ Gráfico de comparação filtrado salvo em: {output_path}")

        except Exception as e:
            print(f"  ❌ ERRO ao processar o arquivo {ppg_filename}: {e}")
            
    print("\n--- Plotagem Final Concluída ---")