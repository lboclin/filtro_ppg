import os
import numpy as np
import pandas as pd
from scipy.signal import detrend

# --- 1. Configurações e Parâmetros ---
# --- Diretórios ---
FILTERED_PPG_DIR = "data/dataset_physionet/filtered_1_ppg/"
FILTERED_IMU_DIR = "data/dataset_physionet/filtered_1_imu/"
RESULTS_DIR = "results/filtro_NLMS/"

# --- Parâmetros do Filtro Adaptativo (NLMS) ---
# Estes são os parâmetros que sintonizamos anteriormente
NLMS_FILTER_ORDER = 15
NLMS_LEARNING_RATE = 0.01 # Usando o valor intermediário que teve algum efeito

# --- Parâmetros da Análise de BPM (FFT) ---
FS = 500.0
WINDOW_SECONDS, STEP_SECONDS = 8, 1
FFT_LENGTH = 4096
BPM_HZ_MIN, BPM_HZ_MAX = 0.8, 4.0 # 48 a 240 BPM

# --- 2. Funções Auxiliares ---

def adaptive_nlms_filter(primary_signal, reference_signal, filter_order, learning_rate):
    """
    Aplica um filtro adaptativo LMS Normalizado (NLMS) para remover o ruído de referência
    do sinal primário. Retorna o sinal limpo (que é o sinal de erro).
    """
    n_samples = len(primary_signal)
    weights = np.zeros(filter_order)
    cleaned_signal = np.zeros(n_samples)
    epsilon = 1e-6  # Para evitar divisão por zero

    for i in range(filter_order, n_samples):
        ref_window = reference_signal[i - filter_order : i][::-1]
        
        # Estima o ruído no sinal PPG
        predicted_noise = np.dot(weights, ref_window)
        
        # O sinal limpo é o sinal original menos o ruído estimado
        error = primary_signal[i] - predicted_noise
        cleaned_signal[i] = error
        
        # Calcula a energia da janela de referência para normalizar o passo
        energy = np.sum(ref_window**2)
        
        # A atualização dos pesos é normalizada pela energia do movimento
        step_size = learning_rate / (energy + epsilon)
        weights = weights + step_size * error * ref_window
        
    return cleaned_signal

def get_bpm_from_fft(window_data, fs):
    """Calcula o BPM a partir de uma janela de sinal usando FFT."""
    if not np.all(np.isfinite(window_data)) or len(window_data) == 0:
        return None
        
    data_detrended = detrend(window_data)
    hanning_win = np.hanning(len(data_detrended))
    data_final = data_detrended * hanning_win
    
    yf = np.abs(np.fft.rfft(data_final, n=FFT_LENGTH))
    xf = np.fft.rfftfreq(FFT_LENGTH, 1 / fs)
    
    freq_mask = (xf >= BPM_HZ_MIN) & (xf <= BPM_HZ_MAX)
    
    if np.any(freq_mask) and np.max(yf[freq_mask]) > 0:
        peak_freq = xf[freq_mask][np.argmax(yf[freq_mask])]
        bpm = peak_freq * 60
        return bpm
    return None

# --- 3. Execução Principal ---
if __name__ == "__main__":
    # Cria a pasta de resultados se não existir
    nlms_results_dir = os.path.join(RESULTS_DIR, "nlms_results")
    os.makedirs(nlms_results_dir, exist_ok=True)
    print(f"Diretório para resultados do NLMS: {os.path.abspath(nlms_results_dir)}")

    ppg_files = sorted([f for f in os.listdir(FILTERED_PPG_DIR) if f.endswith("_filtered_c5.npy")])
    print(f"\nIniciando processamento com o método NLMS para {len(ppg_files)} registros...")

    for ppg_filename in ppg_files:
        base_name = ppg_filename.replace('_filtered_c5.npy', '')
        print(f"\n--- Processando registro: {base_name} ---")

        ppg_path = os.path.join(FILTERED_PPG_DIR, ppg_filename)
        imu_filename = f"{base_name}_imu.npy"
        imu_path = os.path.join(FILTERED_IMU_DIR, imu_filename)

        if not os.path.exists(imu_path):
            print(f"  ℹ️  Arquivo IMU não encontrado para {base_name}. Pulando.")
            continue

        # Carrega os dados pré-filtrados
        ppg_filtered = np.load(ppg_path)
        imu_filtered_3_eixos = np.load(imu_path)
        
        # Calcula a magnitude do movimento
        imu_magnitude = np.sqrt(np.sum(imu_filtered_3_eixos**2, axis=1))

        # Aplica o filtro adaptativo NLMS para limpar o sinal PPG
        ppg_final_limpo = adaptive_nlms_filter(ppg_filtered, imu_magnitude, NLMS_FILTER_ORDER, NLMS_LEARNING_RATE)
        
        # Calcula o BPM no sinal limpo usando janelas deslizantes
        window_samples = int(WINDOW_SECONDS * FS)
        step_samples = int(STEP_SECONDS * FS)
        bpm_results = []

        for i in range(0, len(ppg_final_limpo) - window_samples, step_samples):
            window_start_s = i / FS
            ppg_window = ppg_final_limpo[i : i + window_samples]
            bpm = get_bpm_from_fft(ppg_window, FS)
            
            if bpm is not None:
                bpm_results.append({'tempo_s': window_start_s, 'bpm': bpm})

        if not bpm_results:
            print(f"  ℹ️  Nenhum BPM válido pôde ser calculado para {base_name}.")
            continue
            
        # Salva os resultados em um CSV
        results_df = pd.DataFrame(bpm_results)
        csv_path = os.path.join(nlms_results_dir, f"{base_name}_bpm_results_nlms.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"  ✅ Resultados do NLMS salvos em: {csv_path}")

    print("\n--- Processamento NLMS Concluído ---")