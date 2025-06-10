# -*- coding: utf-8 -*-
"""
Script de Visualização para Sinais Brutos (Pré-filtragem).

Este script carrega os dados brutos de PPG e IMU, que foram extraídos
do formato WFDB e salvos como arquivos .npy. Ele gera gráficos para
cada sinal, permitindo uma inspeção visual inicial da qualidade e
características dos dados antes de qualquer etapa de filtragem.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configurações ---

# --- Diretórios de Entrada ---
# Caminho para os dados de PPG brutos (mas já com a média dos canais calculada).
PRE_FILTERED_PPG_DIR = "data/dataset_physionet/pre_filtered_ppg/"
# Caminho para os dados de IMU brutos (3 eixos).
PRE_FILTERED_IMU_DIR = "data/dataset_physionet/pre_filtered_imu/"

# --- Diretório de Saída ---
# Pasta onde os gráficos dos sinais brutos serão salvos.
PLOT_OUTPUT_DIR = "outputs/output_pre_filtered/"

# --- Parâmetros de Visualização ---
# Define o número de amostras a serem plotadas para um "zoom" no início do sinal.
# Defina como None para plotar o sinal completo.
SAMPLES_TO_PLOT = 10000 
# Frequência de amostragem (Hz) para criar o eixo do tempo em segundos.
FS = 500.0

def plot_ppg_signal(signal_data, base_filename, output_dir):
    """
    Plota e salva o gráfico para um sinal PPG (1D) bruto.

    Args:
        signal_data (np.array): O vetor de dados do sinal PPG.
        base_filename (str): O nome base do registro para o título e nome do arquivo.
        output_dir (str): O diretório onde o gráfico será salvo.
    """
    # Lógica para "zooming": fatia o sinal se SAMPLES_TO_PLOT for definido.
    if SAMPLES_TO_PLOT is not None and SAMPLES_TO_PLOT > 0 and len(signal_data) > SAMPLES_TO_PLOT:
        signal_to_plot = signal_data[:SAMPLES_TO_PLOT]
        zoom_info = f"(Primeiras {len(signal_to_plot)} Amostras)"
    else:
        signal_to_plot = signal_data
        zoom_info = "(Sinal Completo)"
    
    # Cria um eixo de tempo em segundos.
    time_axis = np.arange(len(signal_to_plot)) / FS

    # Cria a figura e plota os dados.
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal_to_plot, label='Sinal PPG (Média dos Canais)', color='crimson')
    
    # Configura os detalhes do gráfico.
    plt.title(f'Sinal PPG Pré-filtrado: {base_filename} {zoom_info}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude Bruta')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Salva a figura do gráfico.
    plot_filename = f"{base_filename}_ppg_pre-filtered.png"
    save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(save_path)
        print(f"  ✅ Gráfico PPG salvo em: {plot_filename}")
    except Exception as e:
        print(f"  ❌ ERRO ao salvar o gráfico PPG {save_path}: {e}")
    plt.close()

def plot_imu_signal(signal_data, base_filename, output_dir):
    """
    Plota e salva o gráfico para um sinal IMU (3 eixos) bruto.

    Args:
        signal_data (np.array): O array 2D com os dados do IMU (n_amostras, 3 eixos).
        base_filename (str): O nome base do registro.
        output_dir (str): O diretório onde o gráfico será salvo.
    """
    # Valida o formato dos dados do IMU.
    if signal_data.ndim != 2 or signal_data.shape[1] != 3:
        print(f"  AVISO: Sinal IMU para {base_filename} não tem 3 eixos (shape: {signal_data.shape}). Pulando plotagem.")
        return

    # Lógica de "zooming".
    if SAMPLES_TO_PLOT is not None and SAMPLES_TO_PLOT > 0 and len(signal_data) > SAMPLES_TO_PLOT:
        signal_to_plot = signal_data[:SAMPLES_TO_PLOT]
        zoom_info = f"(Primeiras {len(signal_to_plot)} Amostras)"
    else:
        signal_to_plot = signal_data
        zoom_info = "(Sinal Completo)"

    time_axis = np.arange(len(signal_to_plot)) / FS

    # Cria a figura e plota cada um dos 3 eixos.
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal_to_plot[:, 0], label='Eixo X', color='royalblue', alpha=0.9)
    plt.plot(time_axis, signal_to_plot[:, 1], label='Eixo Y', color='forestgreen', alpha=0.9)
    plt.plot(time_axis, signal_to_plot[:, 2], label='Eixo Z', color='darkorange', alpha=0.9)
    
    plt.title(f'Sinal IMU Pré-filtrado: {base_filename} {zoom_info}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Aceleração (g)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Salva a figura.
    plot_filename = f"{base_filename}_imu_pre-filtered.png"
    save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(save_path)
        print(f"  ✅ Gráfico IMU salvo em: {plot_filename}")
    except Exception as e:
        print(f"  ❌ ERRO ao salvar o gráfico IMU {save_path}: {e}")
    plt.close()


# --- 3. Execução Principal ---
if __name__ == "__main__":
    # Valida e cria os diretórios necessários.
    if not os.path.isdir(PRE_FILTERED_PPG_DIR):
        print(f"🚨 ERRO: Diretório de entrada PPG não encontrado: '{os.path.abspath(PRE_FILTERED_PPG_DIR)}'")
        exit()
        
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Diretório de saída para os gráficos: {os.path.abspath(PLOT_OUTPUT_DIR)}")

    # O loop principal é guiado pelos arquivos de PPG.
    ppg_files = sorted([f for f in os.listdir(PRE_FILTERED_PPG_DIR) if f.endswith("_ppg.npy")])
    
    print(f"\n🔍 {len(ppg_files)} arquivos de sinal PPG encontrados para plotar.")

    # Itera sobre cada arquivo PPG.
    for ppg_filename in ppg_files:
        # Extrai o nome base para poder encontrar o arquivo IMU correspondente.
        base_name = ppg_filename.replace('_ppg.npy', '')
        print(f"\n--- Processando registro: {base_name} ---")

        # Plota o sinal PPG.
        try:
            ppg_path = os.path.join(PRE_FILTERED_PPG_DIR, ppg_filename)
            ppg_signal = np.load(ppg_path)
            plot_ppg_signal(ppg_signal, base_name, PLOT_OUTPUT_DIR)
        except Exception as e:
            print(f"  ❌ ERRO ao processar o arquivo PPG {ppg_filename}: {e}")
            continue

        # Procura e plota o sinal IMU correspondente.
        imu_filename = f"{base_name}_imu.npy"
        imu_path = os.path.join(PRE_FILTERED_IMU_DIR, imu_filename)
        
        if os.path.exists(imu_path):
            try:
                imu_signal = np.load(imu_path)
                plot_imu_signal(imu_signal, base_name, PLOT_OUTPUT_DIR)
            except Exception as e:
                print(f"  ❌ ERRO ao processar o arquivo IMU {imu_filename}: {e}")
        else:
            print(f"  ℹ️  Arquivo IMU correspondente não encontrado para {base_name}.")

    print(f"\n--- Plotagem Concluída ---")