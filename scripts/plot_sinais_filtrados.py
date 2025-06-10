# -*- coding: utf-8 -*-
"""
Script de Visualização para Sinais Pós-Filtragem.

Este script carrega os dados de PPG e IMU que já passaram pela primeira
etapa de filtragem (pré-processamento) e gera gráficos para cada um.

O objetivo é permitir uma inspeção visual da qualidade dos sinais antes que
eles entrem no algoritmo principal de cálculo de BPM, garantindo que a remoção
de ruído inicial foi bem-sucedida.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configurações ---

# --- Diretórios de Entrada ---
# Caminho para a pasta com os dados de PPG que passaram pelo filtro passa-faixa.
FILTERED_PPG_DIR = "data/dataset_physionet/filtered_1_ppg/"
# Caminho para a pasta com os dados de IMU que passaram pelo filtro passa-baixa.
FILTERED_IMU_DIR = "data/dataset_physionet/filtered_1_imu/" 

# --- Diretório de Saída ---
# Pasta onde os gráficos dos sinais filtrados serão salvos.
PLOT_OUTPUT_DIR = "outputs/output_filtered_1/"

# --- Parâmetros de Visualização ---
# Define o número de amostras a serem plotadas para um "zoom" no início do sinal.
# Defina como None para plotar o sinal completo.
SAMPLES_TO_PLOT = 10000 
# Frequência de amostragem (Hz) para criar o eixo do tempo em segundos.
FS = 500.0

def plot_filtered_ppg(signal_data, base_filename, output_dir):
    """
    Plota e salva o gráfico para um sinal PPG (1D) que já foi filtrado.

    Args:
        signal_data (np.array): O vetor de dados do sinal PPG.
        base_filename (str): O nome base do registro para o título e nome do arquivo.
        output_dir (str): O diretório onde o gráfico será salvo.
    """
    # Lógica para "zooming": fatia o sinal se SAMPLES_TO_PLOT for definido.
    if SAMPLES_TO_PLOT and len(signal_data) > SAMPLES_TO_PLOT:
        signal_to_plot = signal_data[:SAMPLES_TO_PLOT]
        zoom_info = f"(Primeiras {len(signal_to_plot)} Amostras)"
    else:
        signal_to_plot = signal_data
        zoom_info = "(Sinal Completo)"
    
    # Cria um eixo de tempo em segundos dividindo o número de amostras pela frequência de amostragem.
    time_axis = np.arange(len(signal_to_plot)) / FS

    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal_to_plot, label='Sinal PPG Pós-filtro', color='teal')
    
    plt.title(f'Sinal PPG Filtrado (Passa-Faixa): {base_filename} {zoom_info}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude Filtrada')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Salva a figura do gráfico no diretório de saída.
    plot_filename = f"{base_filename}_ppg_filtered.png"
    save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(save_path)
        print(f"  ✅ Gráfico PPG Filtrado salvo em: {plot_filename}")
    except Exception as e:
        print(f"  ❌ ERRO ao salvar o gráfico PPG {save_path}: {e}")
    plt.close() # Libera a memória da figura.

def plot_filtered_imu(signal_data, base_filename, output_dir):
    """
    Plota e salva o gráfico para um sinal IMU (3 eixos) que já foi filtrado.

    Args:
        signal_data (np.array): O array 2D com os dados do IMU (n_amostras, 3 eixos).
        base_filename (str): O nome base do registro para o título e nome do arquivo.
        output_dir (str): O diretório onde o gráfico será salvo.
    """
    # Valida se os dados do IMU têm o formato esperado (3 colunas/eixos).
    if signal_data.ndim != 2 or signal_data.shape[1] != 3:
        print(f"  AVISO: Sinal IMU para {base_filename} não tem 3 eixos. Pulando.")
        return

    # Lógica de "zooming" similar à do PPG.
    if SAMPLES_TO_PLOT and len(signal_data) > SAMPLES_TO_PLOT:
        signal_to_plot = signal_data[:SAMPLES_TO_PLOT]
        zoom_info = f"(Primeiras {len(signal_to_plot)} Amostras)"
    else:
        signal_to_plot = signal_data
        zoom_info = "(Sinal Completo)"

    time_axis = np.arange(len(signal_to_plot)) / FS

    plt.figure(figsize=(15, 5))
    # Plota cada coluna do array (eixo) como uma linha separada no mesmo gráfico.
    plt.plot(time_axis, signal_to_plot[:, 0], label='Eixo X', color='royalblue')
    plt.plot(time_axis, signal_to_plot[:, 1], label='Eixo Y', color='forestgreen')
    plt.plot(time_axis, signal_to_plot[:, 2], label='Eixo Z', color='darkorange')
    
    plt.title(f'Sinal IMU Filtrado (Passa-Baixa): {base_filename} {zoom_info}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Aceleração Filtrada (g)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plot_filename = f"{base_filename}_imu_filtered.png"
    save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(save_path)
        print(f"  ✅ Gráfico IMU Filtrado salvo em: {plot_filename}")
    except Exception as e:
        print(f"  ❌ ERRO ao salvar o gráfico IMU {save_path}: {e}")
    plt.close()


# --- 3. Execução Principal ---
if __name__ == "__main__":
    # Validação dos diretórios de entrada.
    if not os.path.isdir(FILTERED_PPG_DIR):
        print(f"🚨 ERRO: Diretório de entrada PPG não encontrado: '{os.path.abspath(FILTERED_PPG_DIR)}'")
        exit()
        
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Diretório de saída para os gráficos: {os.path.abspath(PLOT_OUTPUT_DIR)}")

    # Usa os arquivos de PPG como guia principal para o loop.
    ppg_files = sorted([f for f in os.listdir(FILTERED_PPG_DIR) if f.endswith("_filtered_c5.npy")])
    
    print(f"\n🔍 {len(ppg_files)} arquivos de sinal PPG filtrado encontrados para plotar.")

    for ppg_filename in ppg_files:
        
        # Extrai o nome base do arquivo para poder encontrar o arquivo IMU correspondente.
        # Ex: "s1_run_filtered_c5.npy" -> "s1_run"
        base_name = ppg_filename.replace('_filtered_c5.npy', '')
        print(f"\n--- Processando registro: {base_name} ---")

        # Processa e plota o sinal PPG.
        try:
            ppg_path = os.path.join(FILTERED_PPG_DIR, ppg_filename)
            ppg_signal = np.load(ppg_path)
            plot_filtered_ppg(ppg_signal, base_name, PLOT_OUTPUT_DIR)
        except Exception as e:
            print(f"  ❌ ERRO ao processar o arquivo PPG {ppg_filename}: {e}")
            continue

        # Encontra, carrega e plota o sinal IMU correspondente.
        imu_filename = f"{base_name}_imu.npy"
        imu_path = os.path.join(FILTERED_IMU_DIR, imu_filename)
        
        if os.path.exists(imu_path):
            try:
                imu_signal = np.load(imu_path)
                plot_filtered_imu(imu_signal, base_name, PLOT_OUTPUT_DIR)
            except Exception as e:
                print(f"  ❌ ERRO ao processar o arquivo IMU {imu_filename}: {e}")
        else:
            print(f"  ℹ️  Arquivo IMU filtrado correspondente não encontrado em '{FILTERED_IMU_DIR}'.")

    print(f"\n--- Plotagem dos Sinais Filtrados Concluída ---")