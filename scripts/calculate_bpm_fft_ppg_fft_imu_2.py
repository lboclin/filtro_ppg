import os
import numpy as np
import pandas as pd
from scipy.signal import detrend # <-- CORREÇÃO: 'hanning' foi REMOVIDO desta linha

# --- 1. Configurações e Parâmetros ---
FILTERED_PPG_DIR = "data/dataset_physionet/filtered_1_ppg/"
FILTERED_IMU_DIR = "data/dataset_physionet/filtered_1_imu/"
RESULTS_DIR = "results/fftPpg_com_valdidadorFftImu_2/"

# --- Parâmetros Gerais ---
FS = 500.0
WINDOW_SECONDS, STEP_SECONDS = 8, 1
FFT_LENGTH = 4096

# --- Parâmetros de Análise ---
BPM_HZ_MIN, BPM_HZ_MAX = 0.8, 3.0  # 48 a 180 BPM para o coração
MOTION_HZ_MIN, MOTION_HZ_MAX = 1.0, 3.5 # 60 a 210 SPM para o movimento

# --- Parâmetros do "Detector de Mentiras" ---
COLLISION_THRESHOLD_HZ = 0.2 
POWER_RATIO_THRESHOLD = 2.0

# --- 2. Função Auxiliar Modificada ---
def get_dominant_freq_and_power(data, fs, min_hz, max_hz):
    """Encontra a frequência dominante e a magnitude do pico em um sinal."""
    if len(data) == 0:
        return None, None
        
    data_detrended = detrend(data)
    
    # <-- CORREÇÃO: A chamada da função 'hanning' agora usa o 'np' do NumPy -->
    hanning_win = np.hanning(len(data_detrended))
    data_final = data_detrended * hanning_win
    
    yf = np.abs(np.fft.rfft(data_final, n=FFT_LENGTH))
    xf = np.fft.rfftfreq(FFT_LENGTH, 1 / fs)
    
    freq_mask = (xf >= min_hz) & (xf <= max_hz)
    
    if np.any(freq_mask) and np.max(yf[freq_mask]) > 0:
        peak_idx_in_mask = np.argmax(yf[freq_mask])
        peak_freq = xf[freq_mask][peak_idx_in_mask]
        peak_power = yf[freq_mask][peak_idx_in_mask]
        return peak_freq, peak_power
    return None, None

# --- 3. Execução Principal ---
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Diretório para resultados CSV: {os.path.abspath(RESULTS_DIR)}")

    ppg_files = sorted([f for f in os.listdir(FILTERED_PPG_DIR) if f.endswith("_filtered_c5.npy")])
    print(f"\nIniciando processamento final de {len(ppg_files)} registros...")

    for ppg_filename in ppg_files:
        base_name = ppg_filename.replace('_filtered_c5.npy', '')
        print(f"\n--- Processando registro: {base_name} ---")

        ppg_path = os.path.join(FILTERED_PPG_DIR, ppg_filename)
        imu_filename = f"{base_name}_imu.npy"
        imu_path = os.path.join(FILTERED_IMU_DIR, imu_filename)

        has_imu = os.path.exists(imu_path)
        ppg_filtered = np.load(ppg_path)
        if has_imu:
            imu_filtered = np.load(imu_path)

        window_samples = int(WINDOW_SECONDS * FS); step_samples = int(STEP_SECONDS * FS)
        final_results = []

        for i in range(0, len(ppg_filtered) - window_samples, step_samples):
            window_start_s = i / FS
            ppg_window = ppg_filtered[i : i + window_samples]
            
            ppg_freq, ppg_power = get_dominant_freq_and_power(ppg_window, FS, BPM_HZ_MIN, BPM_HZ_MAX)
            if ppg_freq is None: continue
            
            bpm_candidate = ppg_freq * 60
            
            is_artifact = False
            if has_imu:
                imu_window = imu_filtered[i : i + window_samples]
                imu_magnitude = np.sqrt(np.sum(imu_window**2, axis=1))
                motion_freq, motion_power = get_dominant_freq_and_power(imu_magnitude, FS, MOTION_HZ_MIN, MOTION_HZ_MAX)
                
                if motion_freq and motion_power:
                    # Lógica de desempate
                    if abs(ppg_freq - motion_freq) < COLLISION_THRESHOLD_HZ:
                        if motion_power * POWER_RATIO_THRESHOLD > ppg_power:
                            is_artifact = True
                    
                    if abs(ppg_freq - 2 * motion_freq) < COLLISION_THRESHOLD_HZ:
                        if motion_power * POWER_RATIO_THRESHOLD > ppg_power:
                            is_artifact = True
            
            if not is_artifact:
                final_results.append({'tempo_s': window_start_s, 'bpm': bpm_candidate})

        if not final_results:
            print("  ℹ️  Nenhuma janela com BPM validado foi encontrada.")
            continue
            
        results_df = pd.DataFrame(final_results)
        csv_path = os.path.join(RESULTS_DIR, f"{base_name}_bpm_results_final_v2.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"  ✅ Resultados de BPM finais (v2) salvos em: {csv_path}")

    print("\n--- Processamento Final Concluído ---")