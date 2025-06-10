import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks # 'hanning' foi removido desta linha

# --- 1. Configurações e Parâmetros ---
FILTERED_PPG_DIR = "data/dataset_physionet/filtered_1_ppg/"
FILTERED_IMU_DIR = "data/dataset_physionet/filtered_1_imu/"
RESULTS_DIR = "results/peaks_filter"

FS = 500.0
BPM_MIN, BPM_MAX = 50.0, 240.0
MIN_DISTANCE_SAMPLES = int((60.0 / BPM_MAX) * FS)
MIN_PEAK_PROMINENCE = 0.1
WINDOW_SECONDS, STEP_SECONDS = 8, 1
FFT_LENGTH = 4096

# --- 2. Funções Auxiliares ---

def get_motion_freq(imu_window, fs):
    """Calcula a frequência dominante do movimento em uma janela do IMU usando FFT."""
    imu_magnitude = np.sqrt(np.sum(imu_window**2, axis=1))
    
    imu_magnitude_detrended = imu_magnitude - np.mean(imu_magnitude)
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Chamando a função hanning a partir do NumPy (np)
    hanning_win = np.hanning(len(imu_magnitude_detrended))
    imu_window_final = imu_magnitude_detrended * hanning_win
    
    yf = np.abs(np.fft.rfft(imu_window_final, n=FFT_LENGTH))
    xf = np.fft.rfftfreq(FFT_LENGTH, 1 / fs)
    
    motion_freq_mask = (xf >= 1.0) & (xf <= 4.0)
    
    if np.any(motion_freq_mask) and np.max(yf[motion_freq_mask]) > 0:
        motion_peak_freq = xf[motion_freq_mask][np.argmax(yf[motion_freq_mask])]
        return motion_peak_freq
    return None

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

        if not os.path.exists(imu_path):
            print("  ℹ️  Arquivo IMU não encontrado. Processando sem correção de movimento."); 
            has_imu = False
        else:
            has_imu = True
            imu_filtered = np.load(imu_path)

        ppg_filtered = np.load(ppg_path)
        
        window_samples = int(WINDOW_SECONDS * FS); step_samples = int(STEP_SECONDS * FS)
        final_results = []

        for i in range(0, len(ppg_filtered) - window_samples, step_samples):
            window_start_s = i / FS
            ppg_window = ppg_filtered[i : i + window_samples]
            
            peaks, _ = find_peaks(ppg_window, distance=MIN_DISTANCE_SAMPLES, prominence=MIN_PEAK_PROMINENCE)
            if len(peaks) < 2: continue
            
            instant_bpms = 60.0 / (np.diff(peaks) / FS)
            candidate_bpms = [bpm for bpm in instant_bpms if BPM_MIN <= bpm <= BPM_MAX]
            if not candidate_bpms: continue

            motion_bpm = None
            if has_imu:
                imu_window = imu_filtered[i : i + window_samples]
                motion_freq = get_motion_freq(imu_window, FS)
                if motion_freq:
                    motion_bpm = motion_freq * 60
            
            validated_bpms = []
            if motion_bpm:
                exclusion_threshold = 15.0
                for bpm in candidate_bpms:
                    if abs(bpm - motion_bpm) > exclusion_threshold:
                        validated_bpms.append(bpm)
            else:
                validated_bpms = candidate_bpms
            
            if len(validated_bpms) >= 2:
                median_bpm = np.median(validated_bpms)
                final_results.append({'tempo_s': window_start_s, 'bpm': median_bpm})

        if not final_results:
            print("  ℹ️  Nenhuma janela com batimentos suficientes para calcular o BPM final.")
            continue
            
        results_df = pd.DataFrame(final_results)
        csv_path = os.path.join(RESULTS_DIR, f"{base_name}_bpm_results_final.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"  ✅ Resultados de BPM finais salvos em: {csv_path}")

    print("\n--- Processamento Final Concluído ---")