# -*- coding: utf-8 -*-
"""
Script de Pré-processamento para Sinais de IMU (Acelerômetro).

Este script carrega os dados brutos de 3 eixos do acelerômetro, aplica um
filtro passa-baixa suave para remover ruído de alta frequência do sensor
(jitter), e salva os dados suavizados em um novo diretório para uso posterior.
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt

# --- 1. Configurações ---

# Diretório onde os dados brutos do IMU (.npy) estão localizados.
INPUT_IMU_DIR = "data/dataset_physionet/pre_filtered_imu/"
# Diretório onde os dados do IMU já filtrados (suavizados) serão salvos.
PROCESSED_IMU_DIR = "data/dataset_physionet/filtered_1_imu/"

# --- Parâmetros do Filtro Passa-Baixa ---
# A frequência de amostragem dos sinais é de 500 Hz.
FS = 500.0
# Frequência de corte para o filtro passa-baixas (em Hz).
# Um valor como 20Hz é alto o suficiente para não afetar os sinais de movimento
# reais (caminhada, corrida), mas baixo o suficiente para remover ruído eletrônico.
LOWPASS_CUTOFF = 20.0
# Ordem do filtro: define quão "íngreme" é a atenuação após a frequência de corte.
FILTER_ORDER = 4

def filter_imu_data(raw_imu_data, fs):
    """
    Aplica um filtro passa-baixa Butterworth de fase zero em cada eixo dos dados do IMU.

    Args:
        raw_imu_data (np.array): Array 2D com os dados do IMU (n_amostras, 3 eixos).
        fs (float): A frequência de amostragem do sinal.

    Returns:
        np.array: O array 2D com os dados do IMU filtrados.
    """
    # Projeta o filtro passa-baixa Butterworth, obtendo seus coeficientes (b, a).
    # Normaliza a frequência de corte pela frequência de Nyquist.
    nyquist_freq = 0.5 * fs
    low_cutoff_norm = LOWPASS_CUTOFF / nyquist_freq
    b, a = butter(FILTER_ORDER, low_cutoff_norm, btype='lowpass', analog=False)
    
    # Aplica o filtro aos dados. 'filtfilt' é usado para evitar atraso de fase no sinal.
    # O parâmetro 'axis=0' é crucial: ele garante que o filtro seja aplicado
    # verticalmente, ao longo do tempo, para cada eixo (coluna) de forma independente.
    filtered_imu_data = filtfilt(b, a, raw_imu_data, axis=0)
    
    return filtered_imu_data

# --- 2. Execução Principal ---
if __name__ == "__main__":
    # Garante que o diretório de saída exista.
    os.makedirs(PROCESSED_IMU_DIR, exist_ok=True)
    print(f"Diretório de saída para IMU processado: {os.path.abspath(PROCESSED_IMU_DIR)}")

    # Verifica se o diretório de entrada existe.
    if not os.path.isdir(INPUT_IMU_DIR):
        print(f"🚨 ERRO: Diretório de entrada não encontrado: {os.path.abspath(INPUT_IMU_DIR)}")
        exit()

    # Encontra todos os arquivos de IMU no diretório de entrada.
    imu_files = sorted([f for f in os.listdir(INPUT_IMU_DIR) if f.endswith("_imu.npy")])
    
    if not imu_files:
        print(f"Nenhum arquivo _imu.npy encontrado em {os.path.abspath(INPUT_IMU_DIR)}")
        exit()

    print(f"\nEncontrados {len(imu_files)} arquivos IMU para filtrar.")

    # Itera sobre cada arquivo de IMU encontrado.
    for filename in imu_files:
        print(f"\nProcessando arquivo: {filename}...")
        
        input_path = os.path.join(INPUT_IMU_DIR, filename)
        
        # Carrega os dados brutos do IMU (um array com 3 colunas/eixos).
        raw_imu = np.load(input_path)
        
        # Chama a função para aplicar o filtro passa-baixa.
        filtered_imu = filter_imu_data(raw_imu, FS)
        
        # Salva o novo array com os dados filtrados no diretório de saída.
        output_path = os.path.join(PROCESSED_IMU_DIR, filename)
        np.save(output_path, filtered_imu)
        print(f"  ✅ Sinal IMU filtrado salvo em: {output_path}")

    print("\n--- Filtragem do IMU Concluída ---")