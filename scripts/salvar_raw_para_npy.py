# -*- coding: utf-8 -*-
"""
Script de Extração e Conversão de Dados Brutos.

Este script é a primeira etapa do pipeline de processamento. Ele lê os
registros de dados no formato WFDB (Waveform Database), extrai os canais de
sinal de interesse (PPG e Acelerômetro/IMU), realiza um pré-processamento
básico (média de canais PPG) e salva os sinais limpos no formato .npy,
que é mais eficiente e fácil de usar nos scripts subsequentes.
"""

import os
import wfdb
import numpy as np

# --- 1. Configurações ---

# --- Diretórios ---
# Diretório onde os dados brutos do dataset (formatos .dat, .hea, etc.) estão localizados.
RAW_DATA_DIR = "data/dataset_physionet/raw/"
# Diretório de saída para os arquivos .npy de PPG já processados (com média calculada).
PRE_FILTERED_PPG_DIR = "data/dataset_physionet/pre_filtered_ppg/"
# Diretório de saída para os arquivos .npy de IMU (3 eixos).
PRE_FILTERED_IMU_DIR = "data/dataset_physionet/pre_filtered_imu/"

def load_and_extract_signals(record_name_base, raw_dir):
    """
    Carrega um registro WFDB, extrai e processa os sinais PPG e de Acelerômetro.

    - Para o PPG: encontra todos os canais 'PLETH' e calcula a média entre eles.
    - Para o IMU: encontra os canais 'a_x', 'a_y', 'a_z' e os empilha em um array.

    Args:
        record_name_base (str): O nome base do registro (ex: 's1_run').
        raw_dir (str): O caminho para o diretório de dados brutos.

    Returns:
        tuple: Uma tupla contendo (sinal_ppg_final, dados_imu, fs_real) ou (None, None, None).
    """
    record_path = os.path.join(raw_dir, record_name_base)
    
    try:
        # Carrega o registro completo usando a biblioteca WFDB.
        record = wfdb.rdrecord(record_path)
        signal_data = record.p_signal 
        current_fs = record.fs
        signal_names = record.sig_name

        # Listas para coletar os sinais encontrados.
        ppg_signals_list = []
        acc_signals_list = []

        if signal_names:
            # Dicionário para garantir a ordem correta dos eixos do acelerômetro (X, Y, Z).
            acc_signals_dict = {}

            # Itera sobre todos os canais disponíveis no registro.
            for i, sig_name in enumerate(signal_names):
                # Procura por canais de PPG (qualquer um que contenha 'PLETH').
                if 'PLETH' in sig_name.upper():
                    ppg_signals_list.append(signal_data[:, i])
                
                # Procura por canais de acelerômetro (qualquer um que comece com 'a_').
                if sig_name.lower().startswith('a_'):
                    acc_signals_dict[sig_name.lower()] = signal_data[:, i]

            # Garante a ordem [X, Y, Z] antes de criar o array final do IMU.
            if 'a_x' in acc_signals_dict and 'a_y' in acc_signals_dict and 'a_z' in acc_signals_dict:
                 acc_signals_list = [acc_signals_dict['a_x'], acc_signals_dict['a_y'], acc_signals_dict['a_z']]

        # --- Processamento dos Sinais Coletados ---

        # Processa o(s) sinal(is) PPG.
        final_ppg_signal = None
        if ppg_signals_list:
            print(f"  -> Encontrados {len(ppg_signals_list)} canais PPG. Calculando a média...")
            # Empilha os canais como colunas e calcula a média para obter um sinal único e mais limpo.
            final_ppg_signal = np.mean(np.stack(ppg_signals_list, axis=1), axis=1)

        # Processa o(s) sinal(is) do acelerômetro.
        acc_data = None
        if acc_signals_list:
            # Empilha os 3 eixos em um único array NumPy de shape (n_amostras, 3).
            acc_data = np.stack(acc_signals_list, axis=1)
            print(f"  -> Sinais de acelerômetro (X,Y,Z) combinados em um array de shape: {acc_data.shape}")

        return final_ppg_signal, acc_data, current_fs

    except Exception as e:
        print(f"  ERRO ao processar o registro {record_name_base}: {e}")
        return None, None, None

# --- 3. Execução Principal ---
if __name__ == "__main__":
    # Garante que os diretórios de saída existam, criando-os se necessário.
    os.makedirs(PRE_FILTERED_PPG_DIR, exist_ok=True)
    os.makedirs(PRE_FILTERED_IMU_DIR, exist_ok=True)
    print(f"Diretório de saída para PPG .npy: {os.path.abspath(PRE_FILTERED_PPG_DIR)}")
    print(f"Diretório de saída para IMU .npy: {os.path.abspath(PRE_FILTERED_IMU_DIR)}")
    
    # Encontra todos os registros no diretório raw procurando por arquivos de cabeçalho '.hea'.
    record_names = sorted([os.path.splitext(f)[0] for f in os.listdir(RAW_DATA_DIR) if f.endswith(".hea")])
    if not record_names:
        print("Nenhum registro encontrado.")
        exit()

    print(f"\nEncontrados {len(record_names)} registros para processar.")

    # Itera sobre cada registro, extrai os sinais e salva-os.
    for rec_name in record_names:
        print(f"\nProcessando registro: {rec_name}...")
        final_ppg, raw_acc, actual_fs = load_and_extract_signals(rec_name, RAW_DATA_DIR)
        
        # Bloco para salvar o sinal PPG, se ele foi encontrado.
        if final_ppg is not None:
            output_filename_ppg = f"{rec_name}_ppg.npy"
            save_path_ppg = os.path.join(PRE_FILTERED_PPG_DIR, output_filename_ppg)
            np.save(save_path_ppg, final_ppg)
            print(f"  ✅ Sinal PPG (média) salvo em: {save_path_ppg} (fs: {actual_fs} Hz)")
        else:
            print(f"  ℹ️  Sinal PPG não encontrado no registro {rec_name}.")

        # Bloco para salvar o sinal IMU, se ele foi encontrado.
        if raw_acc is not None:
            output_filename_acc = f"{rec_name}_imu.npy"
            save_path_acc = os.path.join(PRE_FILTERED_IMU_DIR, output_filename_acc)
            np.save(save_path_acc, raw_acc)
            print(f"  ✅ Sinal IMU (3 eixos) salvo em: {save_path_acc} (fs: {actual_fs} Hz)")
        else:
            print(f"  ℹ️  Sinal de Acelerômetro não encontrado no registro {rec_name}.")

    print("\n--- Processamento Concluído ---")