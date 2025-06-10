# -*- coding: utf-8 -*-
"""
Script de Pré-processamento para Sinais PPG.

Este script carrega os dados brutos de PPG a partir de registros WFDB,
aplica um filtro passa-faixa (band-pass) para isolar a faixa de frequência
de interesse dos batimentos cardíacos, e salva o sinal filtrado em um
arquivo .npy para as próximas etapas de análise.
"""

import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

# --- 1. Configurações ---

# --- Diretórios ---
# Diretório onde os dados brutos no formato WFDB estão localizados.
RAW_DATA_DIR = "data/dataset_physionet/raw/"
# Diretório onde os dados de PPG já filtrados serão salvos.
FILTERED_DATA_DIR = "data/dataset_physionet/filtered_1_ppg/"

# --- Parâmetros dos Filtros ---
# Frequência de amostragem esperada dos sinais.
FS = 500.0
# Frequência de corte para o filtro passa-altas (High-pass).
# Remove flutuações muito lentas (ex: respiração, movimento do corpo).
HIGHPASS_CUTOFF = 0.5
# Frequência de corte para o filtro passa-baixas (Low-pass).
# Remove ruídos de alta frequência (ex: ruído eletrônico).
LOWPASS_CUTOFF = 5.0
# Ordem dos filtros: define quão "íngreme" é a atenuação.
HIGHPASS_FILTER_ORDER = 4
LOWPASS_FILTER_ORDER = 8

def load_and_filter_ppg(record_name_base, raw_dir, fs_expected):
    """
    Carrega um registro WFDB, extrai o sinal PPG, aplica um filtro passa-faixa
    Butterworth, e retorna o sinal filtrado e a frequência de amostragem real.

    Args:
        record_name_base (str): O nome base do registro (ex: 's1_run').
        raw_dir (str): O caminho para o diretório de dados brutos.
        fs_expected (float): A frequência de amostragem esperada para verificação.

    Returns:
        tuple: Uma tupla contendo (sinal_ppg_filtrado, fs_real) ou (None, None).
    """
    record_path = os.path.join(raw_dir, record_name_base)
    
    try:
        # Carrega o registro (sinais e metadados).
        record = wfdb.rdrecord(record_path)
        signal_data = record.p_signal 
        current_fs = record.fs

        # Verifica se a frequência de amostragem do arquivo é a esperada.
        if record.fs != fs_expected:
            print(f"  Atenção: Frequência de amostragem do arquivo ({record.fs} Hz) "
                  f"difere da esperada ({fs_expected} Hz). Usando fs do arquivo.")

        # Encontra o índice do primeiro canal PPG (contendo 'PLETH').
        ppg_channel_index = -1
        if record.sig_name:
            for i, sig_name in enumerate(record.sig_name):
                if 'PLETH' in sig_name.upper():
                    ppg_channel_index = i
                    break # Usa o primeiro canal 'PLETH' que encontrar.
        
        if ppg_channel_index == -1:
            print(f"  ERRO: Canal 'PLETH' não encontrado no registro {record_name_base}.")
            return None, None

        ppg_signal_raw = signal_data[:, ppg_channel_index]

        # --- Aplicação do Filtro Passa-Faixa em duas etapas ---
        
        # Etapa 1: Filtro Passa-Altas (para remover flutuação da linha de base).
        nyquist_freq = 0.5 * current_fs
        high_cutoff_norm = HIGHPASS_CUTOFF / nyquist_freq
        
        # Validação para garantir que a frequência de corte é válida para a fs atual.
        if high_cutoff_norm <= 0 or high_cutoff_norm >= 1:
            print(f"  Aviso: Frequência de corte do passa-altas é inválida. Pulando filtro.")
            ppg_after_hp = ppg_signal_raw
        else:
            b_hp, a_hp = butter(HIGHPASS_FILTER_ORDER, high_cutoff_norm, btype='highpass', analog=False)
            ppg_after_hp = filtfilt(b_hp, a_hp, ppg_signal_raw)

        # Etapa 2: Filtro Passa-Baixas (para remover ruído de alta frequência).
        low_cutoff_norm = LOWPASS_CUTOFF / nyquist_freq
        
        if low_cutoff_norm <= 0 or low_cutoff_norm >= 1:
            print(f"  Aviso: Frequência de corte do passa-baixas é inválida. Pulando filtro.")
            ppg_after_lp = ppg_after_hp
        else:
            b_lp, a_lp = butter(LOWPASS_FILTER_ORDER, low_cutoff_norm, btype='lowpass', analog=False)
            # O filtro passa-baixa é aplicado no sinal que JÁ passou pelo filtro passa-altas.
            ppg_after_lp = filtfilt(b_lp, a_lp, ppg_after_hp)
            
        return ppg_after_lp, current_fs

    except Exception as e:
        print(f"  ERRO ao processar o registro {record_name_base}: {e}")
        return None, None

# --- 3. Execução Principal ---
if __name__ == "__main__":
    os.makedirs(FILTERED_DATA_DIR, exist_ok=True)
    print(f"Diretório de saída para dados filtrados: {os.path.abspath(FILTERED_DATA_DIR)}")

    # Encontra todos os registros a serem processados.
    record_names = sorted([os.path.splitext(f)[0] for f in os.listdir(RAW_DATA_DIR) if f.endswith(".hea")])
    if not record_names:
        print(f"Nenhum arquivo de registro (.hea) encontrado em {os.path.abspath(RAW_DATA_DIR)}")
        exit()

    print(f"Encontrados {len(record_names)} registros para processar.")

    # Itera sobre cada registro, aplicando o filtro e salvando o resultado.
    for rec_name in record_names:
        print(f"\nProcessando registro: {rec_name}...")
        
        # Chama a função principal de carregamento e filtragem.
        filtered_ppg, actual_fs = load_and_filter_ppg(rec_name, RAW_DATA_DIR, FS)
        
        # Salva o sinal filtrado se o processamento foi bem-sucedido.
        if filtered_ppg is not None:
            output_filename = f"{rec_name}_filtered_c5.npy"
            save_path = os.path.join(FILTERED_DATA_DIR, output_filename)
            np.save(save_path, filtered_ppg)
            print(f"  ✅ Sinal filtrado salvo em: {save_path} (fs: {actual_fs} Hz)")
        else:
            print(f"  Não foi possível gerar o sinal filtrado para {rec_name}.")

    print("\n--- Filtragem Concluída ---")