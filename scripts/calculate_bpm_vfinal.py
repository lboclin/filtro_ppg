# -*- coding: utf-8 -*-
"""
Script Principal para Cálculo de Frequência Cardíaca (BPM).

Este script implementa o pipeline final para estimar o BPM a partir de sinais
de PPG, utilizando um sinal de IMU (acelerômetro) para validar e corrigir
os resultados em casos de artefatos de movimento.

A abordagem central é baseada em FFT (Transformada Rápida de Fourier) em janelas
deslizantes, com uma lógica de validação que compara a frequência e a potência
do sinal PPG com a do sinal de movimento para descartar medições não confiáveis.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import detrend

# --- 1. Configurações e Parâmetros ---

# --- Diretórios de Entrada e Saída ---
FILTERED_PPG_DIR = "data/dataset_physionet/filtered_1_ppg/"
FILTERED_IMU_DIR = "data/dataset_physionet/filtered_1_imu/"
RESULTS_DIR = "results/bpm_vfinal/"

# --- Parâmetros Gerais de Processamento ---
FS = 500.0  # Frequência de amostragem dos sinais (em Hz).
WINDOW_SECONDS = 8  # Duração da janela de análise (em segundos).
STEP_SECONDS = 1    # Passo da janela deslizante (em segundos).
FFT_LENGTH = 4096   # Resolução da FFT (deve ser uma potência de 2, maior que o tamanho da janela).

# --- Parâmetros de Análise de Frequência ---
# Define a faixa de frequência onde o algoritmo irá procurar picos.
BPM_HZ_MIN, BPM_HZ_MAX = 0.8, 4.0      # Para o coração: 48 a 240 BPM.
MOTION_HZ_MIN, MOTION_HZ_MAX = 1.0, 3.5  # Para o movimento: 60 a 210 passos/minuto.

# --- Parâmetros do "Detector de Mentiras" (Validação) ---
# Se a frequência do PPG e do IMU estiverem mais próximas que este valor, consideramos uma "colisão".
COLLISION_THRESHOLD_HZ = 0.25  # Equivalente a ~15 BPM.
# Critério de desempate para colisões: o pico do IMU precisa ser X vezes mais forte que o do PPG
# para que o BPM seja classificado como artefato de movimento.
POWER_RATIO_THRESHOLD = 2.0

# --- 2. Função Auxiliar ---
def get_dominant_freq_and_power(data, fs, min_hz, max_hz):
    """
    Encontra a frequência dominante e a magnitude (potência) do pico em um sinal
    usando a análise de FFT dentro de uma faixa de frequência específica.

    Args:
        data (np.array): O vetor de dados do sinal (1D).
        fs (float): A frequência de amostragem.
        min_hz (float): A frequência mínima para a busca do pico.
        max_hz (float): A frequência máxima para a busca do pico.

    Returns:
        tuple: Uma tupla contendo (frequência_do_pico, potência_do_pico) ou (None, None).
    """
    if len(data) < 10: return None, None
    
    # Remove a tendência linear da janela para estabilizar a FFT.
    data_detrended = detrend(data)
    
    # Aplica uma janela de Hanning para reduzir o vazamento espectral.
    hanning_win = np.hanning(len(data_detrended))
    data_final = data_detrended * hanning_win
    
    # Calcula a FFT e o vetor de frequências correspondente.
    yf = np.abs(np.fft.rfft(data_final, n=FFT_LENGTH))
    xf = np.fft.rfftfreq(FFT_LENGTH, 1 / fs)
    
    # Cria uma máscara para buscar o pico apenas na faixa de interesse.
    freq_mask = (xf >= min_hz) & (xf <= max_hz)
    
    # Verifica se há picos válidos na faixa e retorna a frequência e potência do mais forte.
    if np.any(freq_mask) and np.max(yf[freq_mask]) > 0:
        peak_idx_in_mask = np.argmax(yf[freq_mask])
        original_indices = np.where(freq_mask)[0]
        peak_idx_original = original_indices[peak_idx_in_mask]
        
        peak_freq = xf[peak_idx_original]
        peak_power = yf[peak_idx_original]
        return peak_freq, peak_power
        
    return None, None

# --- 3. Execução Principal ---
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Diretório para resultados CSV: {os.path.abspath(RESULTS_DIR)}")
    
    # Encontra todos os arquivos de PPG já filtrados.
    ppg_files = sorted([f for f in os.listdir(FILTERED_PPG_DIR) if f.endswith("_filtered_c5.npy")])
    print(f"\nIniciando processamento final de {len(ppg_files)} registros com desempate por potência...")

    # Itera sobre cada registro para processamento individual.
    for ppg_filename in ppg_files:
        base_name = ppg_filename.replace('_filtered_c5.npy', '')
        print(f"\n--- Processando registro: {base_name} ---")

        # Constrói os caminhos para os arquivos PPG e IMU.
        ppg_path = os.path.join(FILTERED_PPG_DIR, ppg_filename)
        imu_filename = f"{base_name}_imu.npy"
        imu_path = os.path.join(FILTERED_IMU_DIR, imu_filename)

        # Carrega os dados, verificando se o arquivo IMU existe.
        has_imu = os.path.exists(imu_path)
        ppg_filtered = np.load(ppg_path)
        if has_imu:
            imu_filtered = np.load(imu_path)

        window_samples = int(WINDOW_SECONDS * FS)
        step_samples = int(STEP_SECONDS * FS)
        final_results = []

        # Loop da janela deslizante: analisa o sinal em trechos de 8 segundos.
        for i in range(0, len(ppg_filtered) - window_samples, step_samples):
            window_start_s = i / FS
            ppg_window = ppg_filtered[i : i + window_samples]
            
            # Etapa 1: Encontra o BPM candidato e sua potência no sinal PPG.
            ppg_freq, ppg_power = get_dominant_freq_and_power(ppg_window, FS, BPM_HZ_MIN, BPM_HZ_MAX)
            if ppg_freq is None: continue
            
            bpm_candidate = ppg_freq * 60
            
            # Etapa 2: Validação com IMU (se disponível).
            is_artifact = False
            if has_imu:
                imu_window = imu_filtered[i : i + window_samples]
                # Calcula a magnitude do movimento a partir dos 3 eixos.
                imu_magnitude = np.sqrt(np.sum(imu_window**2, axis=1))
                # Encontra a frequência e potência do movimento.
                motion_freq, motion_power = get_dominant_freq_and_power(imu_magnitude, FS, MOTION_HZ_MIN, MOTION_HZ_MAX)
                
                # Aplica a lógica do "Detector de Mentiras" se o movimento foi detectado.
                if motion_freq and motion_power:
                    # Verifica colisão com a frequência fundamental do movimento.
                    if abs(ppg_freq - motion_freq) < COLLISION_THRESHOLD_HZ:
                        # Se colidiram, aplica o desempate pela potência.
                        if motion_power * POWER_RATIO_THRESHOLD > ppg_power:
                            is_artifact = True
                    
                    # Verifica colisão com o primeiro harmônico do movimento (2x a frequência).
                    if abs(ppg_freq - 2 * motion_freq) < COLLISION_THRESHOLD_HZ:
                        if motion_power * POWER_RATIO_THRESHOLD > ppg_power:
                            is_artifact = True
            
            # Etapa 3: Salva o resultado apenas se não for classificado como um artefato.
            if not is_artifact:
                final_results.append({'tempo_s': window_start_s, 'bpm': bpm_candidate})

        # Ao final do processamento do arquivo, salva os resultados válidos em um CSV.
        if not final_results:
            print(f"  ℹ️  Nenhuma janela com BPM validado foi encontrada para {base_name}.")
            continue
            
        results_df = pd.DataFrame(final_results)
        csv_path = os.path.join(RESULTS_DIR, f"{base_name}_bpm_results_final_v_power.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"  ✅ Resultados de BPM (com desempate por potência) salvos em: {csv_path}")

    print("\n--- Processamento Final Concluído ---")