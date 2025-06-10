import os
import wfdb
import numpy as np
import pandas as pd
from biosppy.signals import ecg

# --- 1. Configurações e Parâmetros ---
RAW_DATA_DIR = "data/dataset_physionet/raw/"
GROUND_TRUTH_DIR = "results/ground_truth/"

# --- 2. Execução Principal ---
if __name__ == "__main__":
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    print(f"Diretório de saída para o Ground Truth: {os.path.abspath(GROUND_TRUTH_DIR)}")

    record_names = sorted([os.path.splitext(f)[0] for f in os.listdir(RAW_DATA_DIR) if f.endswith(".hea")])
    if not record_names:
        print(f"🚨 ERRO: Nenhum registro encontrado em '{RAW_DATA_DIR}'")
        exit()

    print(f"\nIniciando geração de Ground Truth para {len(record_names)} registros usando BioSPPy...")

    for rec_name in record_names:
        print(f"\n--- Processando registro: {rec_name} ---")
        
        try:
            record = wfdb.rdrecord(os.path.join(RAW_DATA_DIR, rec_name))
            
            ecg_channel_idx = -1
            for i, sig_name in enumerate(record.sig_name):
                if 'ECG' in sig_name.upper():
                    ecg_channel_idx = i
                    break
            
            if ecg_channel_idx == -1:
                print(f"  ℹ️  Canal ECG não encontrado para {rec_name}. Pulando.")
                continue

            ecg_signal_raw = record.p_signal[:, ecg_channel_idx]
            fs = record.fs

            # --- ETAPA DE NORMALIZAÇÃO ADICIONADA AQUI ---
            # Remove a média e divide pelo desvio padrão
            # Isso garante que o sinal tenha média 0 e desvio padrão 1
            if np.std(ecg_signal_raw) > 0:
                ecg_signal_normalized = (ecg_signal_raw - np.mean(ecg_signal_raw)) / np.std(ecg_signal_raw)
            else:
                ecg_signal_normalized = ecg_signal_raw # Evita divisão por zero se o sinal for plano
            
            # --- Usa o sinal NORMALIZADO para a detecção de picos ---
            ecg_results = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=fs, show=False)
            
            r_peaks_indices = ecg_results['rpeaks']
            
            if len(r_peaks_indices) < 5: # Aumentando a exigência para maior confiança
                print(f"  ℹ️  Número insuficiente de picos R detectados por BioSPPy em {rec_name}. Pulando.")
                continue
            
            print(f"  Encontrados {len(r_peaks_indices)} picos R.")
            
            rr_intervals_s = np.diff(r_peaks_indices) / fs
            ecg_bpms = 60.0 / rr_intervals_s
            ecg_bpm_times_s = r_peaks_indices[1:] / fs
            
            results_df = pd.DataFrame({
                'tempo_s': ecg_bpm_times_s,
                'bpm_ecg': ecg_bpms
            })
            
            csv_path = os.path.join(GROUND_TRUTH_DIR, f"{rec_name}_ecg_bpm.csv")
            results_df.to_csv(csv_path, index=False, float_format='%.2f')
            print(f"  ✅ Ground Truth de BPM salvo em: {csv_path}")

        except Exception as e:
            print(f"  ❌ ERRO ao processar o registro {rec_name}: {e}")

    print("\n--- Geração de Ground Truth Concluída ---")