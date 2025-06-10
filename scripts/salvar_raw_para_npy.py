import os
import wfdb
import numpy as np

# --- Configurações ---
# Caminhos relativos
RAW_DATA_DIR = "data/dataset_physionet/raw/"
PRE_FILTERED_PPG_DIR = "data/dataset_physionet/pre_filtered_ppg/"
PRE_FILTERED_IMU_DIR = "data/dataset_physionet/pre_filtered_imu/"

def load_and_extract_signals(record_name_base, raw_dir):
    """
    Carrega um registro WFDB e extrai os sinais PPG e de Acelerômetro (IMU).
    Se múltiplos canais PPG forem encontrados, eles são calculados em uma média.
    Procura por canais de aceleração 'a_x', 'a_y', 'a_z'.
    """
    record_path = os.path.join(raw_dir, record_name_base)
    
    try:
        record = wfdb.rdrecord(record_path)
        signal_data = record.p_signal 
        current_fs = record.fs
        signal_names = record.sig_name

        ppg_signals_list = []
        acc_signals_list = []

        if signal_names:
            # Lista para garantir a ordem correta dos eixos (X, Y, Z)
            acc_signals_dict = {}

            for i, sig_name in enumerate(signal_names):
                # Procura pelo sinal PPG
                if 'PLETH' in sig_name.upper():
                    ppg_signals_list.append(signal_data[:, i])
                
                # --- MUDANÇA CRÍTICA AQUI ---
                # Procura por sinais de acelerômetro começando com 'a_'
                if sig_name.lower().startswith('a_'):
                    # Armazena no dicionário para garantir a ordem
                    acc_signals_dict[sig_name.lower()] = signal_data[:, i]

            # Garante a ordem X, Y, Z antes de empilhar
            if 'a_x' in acc_signals_dict and 'a_y' in acc_signals_dict and 'a_z' in acc_signals_dict:
                 acc_signals_list = [acc_signals_dict['a_x'], acc_signals_dict['a_y'], acc_signals_dict['a_z']]

        # Processa o(s) sinal(is) PPG
        final_ppg_signal = None
        if ppg_signals_list:
            print(f"   -> Encontrados {len(ppg_signals_list)} canais PPG. Calculando a média...")
            final_ppg_signal = np.mean(np.stack(ppg_signals_list, axis=1), axis=1)

        # Processa o(s) sinal(is) do acelerômetro
        acc_data = None
        if acc_signals_list:
            acc_data = np.stack(acc_signals_list, axis=1)
            print(f"   -> Sinais de acelerômetro (X,Y,Z) combinados em um array de shape: {acc_data.shape}")

        return final_ppg_signal, acc_data, current_fs

    except Exception as e:
        print(f"  ERRO ao processar o registro {record_name_base}: {e}")
        return None, None, None

if __name__ == "__main__":
    # O resto do código principal continua exatamente o mesmo
    os.makedirs(PRE_FILTERED_PPG_DIR, exist_ok=True)
    os.makedirs(PRE_FILTERED_IMU_DIR, exist_ok=True)
    print(f"Diretório de saída para PPG .npy: {os.path.abspath(PRE_FILTERED_PPG_DIR)}")
    print(f"Diretório de saída para IMU .npy: {os.path.abspath(PRE_FILTERED_IMU_DIR)}")
    record_names = sorted([os.path.splitext(f)[0] for f in os.listdir(RAW_DATA_DIR) if f.endswith(".hea")])
    if not record_names:
        print("Nenhum registro encontrado.")
        exit()

    print(f"\nEncontrados {len(record_names)} registros para processar.")

    for rec_name in record_names:
        print(f"\nProcessando registro: {rec_name}...")
        final_ppg, raw_acc, actual_fs = load_and_extract_signals(rec_name, RAW_DATA_DIR)
        
        if final_ppg is not None:
            output_filename_ppg = f"{rec_name}_ppg.npy"
            save_path_ppg = os.path.join(PRE_FILTERED_PPG_DIR, output_filename_ppg)
            np.save(save_path_ppg, final_ppg)
            print(f"  ✅ Sinal PPG (média) salvo em: {save_path_ppg} (fs: {actual_fs} Hz)")
        else:
            print(f"  ℹ️  Sinal PPG não encontrado no registro {rec_name}.")

        if raw_acc is not None:
            output_filename_acc = f"{rec_name}_imu.npy"
            save_path_acc = os.path.join(PRE_FILTERED_IMU_DIR, output_filename_acc)
            np.save(save_path_acc, raw_acc)
            print(f"  ✅ Sinal IMU (3 eixos) salvo em: {save_path_acc} (fs: {actual_fs} Hz)")
        else:
            print(f"  ℹ️  Sinal de Acelerômetro não encontrado no registro {rec_name}.")

    print("\n--- Processamento Concluído ---")