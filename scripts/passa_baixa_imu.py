import os
import numpy as np
from scipy.signal import butter, filtfilt

# --- Configurações ---
# Diretórios de entrada e saída
INPUT_IMU_DIR = "data/dataset_physionet/pre_filtered_imu/"
PROCESSED_IMU_DIR = "data/dataset_physionet/filtered_1_imu/"

# Parâmetros do Filtro Passa-Baixa para suavização do IMU
# A frequência de amostragem é 500 Hz, conforme os arquivos
FS = 500.0
# Frequência de corte para o filtro passa-baixas (Hz)
# Um valor como 20Hz remove o ruído do sensor sem afetar o movimento real.
LOWPASS_CUTOFF = 20.0
# Ordem do filtro
FILTER_ORDER = 4

def filter_imu_data(raw_imu_data, fs):
    """
    Aplica um filtro passa-baixa suave em cada eixo dos dados do IMU.
    """
    # Projeta o filtro passa-baixa Butterworth
    nyquist_freq = 0.5 * fs
    low_cutoff_norm = LOWPASS_CUTOFF / nyquist_freq
    
    b, a = butter(FILTER_ORDER, low_cutoff_norm, btype='lowpass', analog=False)
    
    # Aplica o filtro de fase zero em cada coluna (eixo) do array
    # axis=0 garante que o filtro seja aplicado ao longo do tempo para cada eixo
    filtered_imu_data = filtfilt(b, a, raw_imu_data, axis=0)
    
    return filtered_imu_data

if __name__ == "__main__":
    # Garante que o diretório de saída exista
    os.makedirs(PROCESSED_IMU_DIR, exist_ok=True)
    print(f"Diretório de saída para IMU processado: {os.path.abspath(PROCESSED_IMU_DIR)}")

    # Verifica se o diretório de entrada existe
    if not os.path.isdir(INPUT_IMU_DIR):
        print(f"🚨 ERRO: Diretório de entrada não encontrado: {os.path.abspath(INPUT_IMU_DIR)}")
        exit()

    # Processa todos os arquivos .npy no diretório de entrada
    imu_files = sorted([f for f in os.listdir(INPUT_IMU_DIR) if f.endswith("_imu.npy")])
    
    if not imu_files:
        print(f"Nenhum arquivo _imu.npy encontrado em {os.path.abspath(INPUT_IMU_DIR)}")
        exit()

    print(f"\nEncontrados {len(imu_files)} arquivos IMU para filtrar.")

    for filename in imu_files:
        print(f"\nProcessando arquivo: {filename}...")
        
        input_path = os.path.join(INPUT_IMU_DIR, filename)
        
        # Carrega os dados brutos do IMU (3 eixos)
        raw_imu = np.load(input_path)
        
        # Aplica o filtro passa-baixa
        filtered_imu = filter_imu_data(raw_imu, FS)
        
        # Salva os dados filtrados
        output_path = os.path.join(PROCESSED_IMU_DIR, filename)
        np.save(output_path, filtered_imu)
        print(f"  ✅ Sinal IMU filtrado salvo em: {output_path}")

    print("\n--- Filtragem do IMU Concluída ---")