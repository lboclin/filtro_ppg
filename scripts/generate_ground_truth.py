# -*- coding: utf-8 -*-
"""
Script para Geração de Ground Truth de BPM a partir de Sinais de ECG.

Este script lê os registros brutos do dataset, extrai o canal de ECG,
e utiliza a biblioteca BioSPPy para aplicar um algoritmo robusto de detecção
de picos R (baseado em Pan-Tompkins).

A partir dos picos R detectados, ele calcula os intervalos R-R e a frequência
cardíaca instantânea (BPM), salvando essa série temporal em um arquivo .csv.
Este resultado serve como o "padrão ouro" ou "gabarito" para avaliar o
desempenho do nosso algoritmo de estimativa de BPM a partir do PPG.
"""

import os
import wfdb
import numpy as np
import pandas as pd
from biosppy.signals import ecg

# --- 1. Configurações e Parâmetros ---

# Diretório contendo os dados brutos do dataset no formato WFDB.
RAW_DATA_DIR = "data/dataset_physionet/raw/"
# Diretório onde os arquivos .csv com o BPM do ECG (ground truth) serão salvos.
GROUND_TRUTH_DIR = "results/ground_truth/"

# --- 2. Execução Principal ---
if __name__ == "__main__":
    # Garante que o diretório de saída exista; se não, ele será criado.
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    print(f"Diretório de saída para o Ground Truth: {os.path.abspath(GROUND_TRUTH_DIR)}")

    # Encontra todos os registros no diretório raw, baseando-se nos arquivos de cabeçalho (.hea).
    record_names = sorted([os.path.splitext(f)[0] for f in os.listdir(RAW_DATA_DIR) if f.endswith(".hea")])
    if not record_names:
        print(f"🚨 ERRO: Nenhum registro encontrado em '{RAW_DATA_DIR}'")
        exit()

    print(f"\nIniciando geração de Ground Truth para {len(record_names)} registros usando BioSPPy...")

    # Itera sobre cada registro encontrado para processamento individual.
    for rec_name in record_names:
        print(f"\n--- Processando registro: {rec_name} ---")
        
        try:
            # Carrega o registro completo (sinais e metadados) usando a biblioteca WFDB.
            record = wfdb.rdrecord(os.path.join(RAW_DATA_DIR, rec_name))
            
            # Procura pelo canal de ECG nos nomes de sinais do registro.
            ecg_channel_idx = -1
            for i, sig_name in enumerate(record.sig_name):
                if 'ECG' in sig_name.upper():
                    ecg_channel_idx = i
                    break
            
            # Se nenhum canal de ECG for encontrado, pula para o próximo registro.
            if ecg_channel_idx == -1:
                print(f"  ℹ️  Canal ECG não encontrado para {rec_name}. Pulando.")
                continue

            # Extrai o sinal de ECG bruto e a frequência de amostragem do arquivo.
            ecg_signal_raw = record.p_signal[:, ecg_channel_idx]
            fs = record.fs

            # --- ETAPA DE NORMALIZAÇÃO DO SINAL ---
            # Padroniza o sinal para ter média 0 e desvio padrão 1.
            # Isso torna a detecção de picos mais robusta, pois independe da amplitude
            # ou do offset DC original do sinal.
            if np.std(ecg_signal_raw) > 0:
                ecg_signal_normalized = (ecg_signal_raw - np.mean(ecg_signal_raw)) / np.std(ecg_signal_raw)
            else:
                ecg_signal_normalized = ecg_signal_raw # Evita divisão por zero.
            
            # --- DETECÇÃO DE PICOS R COM BIOSPPY ---
            # Chama a função principal da biblioteca, que executa um algoritmo otimizado.
            # O parâmetro 'show=False' impede que a função tente gerar um gráfico.
            ecg_results = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=fs, show=False)
            
            # Extrai os índices dos picos R detectados do dicionário de resultados.
            r_peaks_indices = ecg_results['rpeaks']
            
            # Validação: exige um número mínimo de picos para gerar um resultado confiável.
            if len(r_peaks_indices) < 5:
                print(f"  ℹ️  Número insuficiente de picos R detectados por BioSPPy em {rec_name}. Pulando.")
                continue
            
            print(f"  Encontrados {len(r_peaks_indices)} picos R.")
            
            # --- CÁLCULO DO BPM ---
            # Calcula a diferença entre picos R consecutivos para obter os intervalos R-R.
            rr_intervals_s = np.diff(r_peaks_indices) / fs
            
            # Converte os intervalos de tempo para batimentos por minuto.
            ecg_bpms = 60.0 / rr_intervals_s
            
            # Cria um vetor de tempo para cada valor de BPM calculado.
            ecg_bpm_times_s = r_peaks_indices[1:] / fs
            
            # --- SALVAMENTO DOS RESULTADOS ---
            # Organiza os resultados em um DataFrame do pandas.
            results_df = pd.DataFrame({
                'tempo_s': ecg_bpm_times_s,
                'bpm_ecg': ecg_bpms
            })
            
            # Salva o DataFrame em um arquivo .csv.
            csv_path = os.path.join(GROUND_TRUTH_DIR, f"{rec_name}_ecg_bpm.csv")
            results_df.to_csv(csv_path, index=False, float_format='%.2f')
            print(f"  ✅ Ground Truth de BPM salvo em: {csv_path}")

        except Exception as e:
            print(f"  ❌ ERRO ao processar o registro {rec_name}: {e}")

    print("\n--- Geração de Ground Truth Concluída ---")