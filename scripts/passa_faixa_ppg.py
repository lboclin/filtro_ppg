import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

# --- Configurações ---
# Caminhos relativos assumindo que o script está em filtro_ppg/scripts/
RAW_DATA_DIR = "data/dataset_physionet/raw/"
FILTERED_DATA_DIR = "data/dataset_physionet/filtered_1_ppg/"

# Parâmetros dos filtros (ajuste conforme necessário)
FS = 500.0  # Frequência de amostragem esperada (Hz) - conforme informado
HIGHPASS_CUTOFF = 0.5  # Frequência de corte para o filtro passa-altas (Hz)
LOWPASS_CUTOFF = 5.0   # Frequência de corte para o filtro passa-baixas (Hz)
HIGHPASS_FILTER_ORDER = 4
LOWPASS_FILTER_ORDER = 8

def load_and_filter_ppg(record_name_base, raw_dir, fs_expected):
    """
    Carrega um registro WFDB, extrai o sinal PPG, aplica filtros
    passa-altas e passa-baixas, e retorna o sinal filtrado e a fs.
    """
    record_path = os.path.join(raw_dir, record_name_base)
    
    try:
        # Carrega o registro (dados e metadados)
        record = wfdb.rdrecord(record_path)
        # Carrega os dados do sinal
        signal_data = record.p_signal 
        
        if record.fs != fs_expected:
            print(f"  Atenção: Frequência de amostragem do arquivo ({record.fs} Hz) "
                  f"difere da esperada ({fs_expected} Hz) para {record_name_base}. Usando fs do arquivo: {record.fs} Hz.")
        current_fs = record.fs # Usa a fs real do arquivo

        # Encontra o canal PPG (procurando por 'PLETH')
        ppg_channel_index = -1
        if record.sig_name:
            for i, sig_name in enumerate(record.sig_name):
                if 'PLETH' in sig_name.upper():
                    ppg_channel_index = i
                    break
        
        if ppg_channel_index == -1:
            print(f"  ERRO: Canal 'PLETH' não encontrado no registro {record_name_base}. Pulando este arquivo.")
            return None, None

        ppg_signal_raw = signal_data[:, ppg_channel_index]

        # 1. Filtro Passa-Altas (para remover baseline wander)
        nyquist_freq_hp = 0.5 * current_fs
        high_cutoff_norm = HIGHPASS_CUTOFF / nyquist_freq_hp
        if high_cutoff_norm <= 0 or high_cutoff_norm >= 1:
             print(f"  Aviso: Frequência de corte do passa-altas ({HIGHPASS_CUTOFF} Hz) é inválida para fs={current_fs} Hz. Pulando filtro passa-altas.")
             ppg_after_hp = ppg_signal_raw # Pula o filtro
        else:
            b_hp, a_hp = butter(HIGHPASS_FILTER_ORDER, high_cutoff_norm, btype='highpass', analog=False)
            ppg_after_hp = filtfilt(b_hp, a_hp, ppg_signal_raw)

        # 2. Filtro Passa-Baixas (para remover ruído de alta frequência)
        nyquist_freq_lp = 0.5 * current_fs
        low_cutoff_norm = LOWPASS_CUTOFF / nyquist_freq_lp
        if low_cutoff_norm <= 0 or low_cutoff_norm >= 1:
            print(f"  Aviso: Frequência de corte do passa-baixas ({LOWPASS_CUTOFF} Hz) é inválida para fs={current_fs} Hz. Pulando filtro passa-baixas.")
            ppg_after_lp = ppg_after_hp # Pula o filtro se o passa-altas também foi pulado, ou usa a saída do passa-altas
        else:
            b_lp, a_lp = butter(LOWPASS_FILTER_ORDER, low_cutoff_norm, btype='lowpass', analog=False)
            ppg_after_lp = filtfilt(b_lp, a_lp, ppg_after_hp)
            
        return ppg_after_lp, current_fs

    except Exception as e:
        print(f"  ERRO ao processar o registro {record_name_base}: {e}")
        return None, None

if __name__ == "__main__":
    # Garante que o diretório de saída exista
    os.makedirs(FILTERED_DATA_DIR, exist_ok=True)
    print(f"Diretório de saída para dados filtrados: {os.path.abspath(FILTERED_DATA_DIR)}")

    # Encontra todos os arquivos de cabeçalho (.hea) para identificar os registros
    record_names = []
    if os.path.isdir(RAW_DATA_DIR):
        for filename in os.listdir(RAW_DATA_DIR):
            if filename.endswith(".hea"):
                record_names.append(os.path.splitext(filename)[0])
    else:
        print(f"🚨 ERRO: Diretório de dados brutos não encontrado: {os.path.abspath(RAW_DATA_DIR)}")
        exit()
        
    if not record_names:
        print(f"Nenhum arquivo de registro (.hea) encontrado em {os.path.abspath(RAW_DATA_DIR)}")
        exit()

    print(f"Encontrados {len(record_names)} registros para processar.")

    for rec_name in record_names:
        print(f"\nProcessando registro: {rec_name}...")
        
        filtered_ppg, actual_fs = load_and_filter_ppg(rec_name, RAW_DATA_DIR, FS)
        
        if filtered_ppg is not None:
            output_filename = f"{rec_name}_filtered_c5.npy"
            save_path = os.path.join(FILTERED_DATA_DIR, output_filename)
            np.save(save_path, filtered_ppg)
            print(f"  ✅ Sinal filtrado salvo em: {save_path} (fs: {actual_fs} Hz)")
        else:
            print(f"  Não foi possível gerar o sinal filtrado para {rec_name}.")

    print("\n--- Filtragem Concluída ---")