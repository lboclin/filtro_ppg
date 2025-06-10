import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configurações ---
# Caminhos de entrada
PRE_FILTERED_PPG_DIR = "data/dataset_physionet/pre_filtered_ppg/"
PRE_FILTERED_IMU_DIR = "data/dataset_physionet/pre_filtered_imu/" # Adicionado diretório do IMU

# Caminho de saída para os gráficos
PLOT_OUTPUT_DIR = "outputs/output_pre_filtered/"

# Número de amostras a serem plotadas (se None, plota tudo)
SAMPLES_TO_PLOT = 10000 
# Frequência de amostragem para o eixo do tempo (assumindo 500 Hz para ambos)
FS = 500.0

def plot_ppg_signal(signal_data, base_filename, output_dir):
    """Plota e salva o gráfico para um sinal PPG (1D)."""
    
    # Define a porção do sinal a ser plotada
    if SAMPLES_TO_PLOT is not None and SAMPLES_TO_PLOT > 0 and len(signal_data) > SAMPLES_TO_PLOT:
        signal_to_plot = signal_data[:SAMPLES_TO_PLOT]
        zoom_info = f"(Primeiras {len(signal_to_plot)} Amostras)"
    else:
        signal_to_plot = signal_data
        zoom_info = "(Sinal Completo)"
    
    time_axis = np.arange(len(signal_to_plot)) / FS

    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal_to_plot, label='Sinal PPG (Média dos Canais)', color='crimson')
    
    plt.title(f'Sinal PPG Pré-filtrado: {base_filename} {zoom_info}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude Bruta')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Salva o gráfico
    plot_filename = f"{base_filename}_ppg_pre-filtered.png"
    save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(save_path)
        print(f"  ✅ Gráfico PPG salvo em: {plot_filename}")
    except Exception as e:
        print(f"  ❌ ERRO ao salvar o gráfico PPG {save_path}: {e}")
    plt.close()

def plot_imu_signal(signal_data, base_filename, output_dir):
    """Plota e salva o gráfico para um sinal IMU (3 eixos)."""
    
    if signal_data.ndim != 2 or signal_data.shape[1] != 3:
        print(f"  AVISO: Sinal IMU para {base_filename} não tem 3 eixos (shape: {signal_data.shape}). Pulando plotagem.")
        return

    # Define a porção do sinal a ser plotada
    if SAMPLES_TO_PLOT is not None and SAMPLES_TO_PLOT > 0 and len(signal_data) > SAMPLES_TO_PLOT:
        signal_to_plot = signal_data[:SAMPLES_TO_PLOT]
        zoom_info = f"(Primeiras {len(signal_to_plot)} Amostras)"
    else:
        signal_to_plot = signal_data
        zoom_info = "(Sinal Completo)"

    time_axis = np.arange(len(signal_to_plot)) / FS

    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal_to_plot[:, 0], label='Eixo X', color='royalblue', alpha=0.9)
    plt.plot(time_axis, signal_to_plot[:, 1], label='Eixo Y', color='forestgreen', alpha=0.9)
    plt.plot(time_axis, signal_to_plot[:, 2], label='Eixo Z', color='darkorange', alpha=0.9)
    
    plt.title(f'Sinal IMU Pré-filtrado: {base_filename} {zoom_info}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Aceleração (g)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Salva o gráfico
    plot_filename = f"{base_filename}_imu_pre-filtered.png"
    save_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(save_path)
        print(f"  ✅ Gráfico IMU salvo em: {plot_filename}")
    except Exception as e:
        print(f"  ❌ ERRO ao salvar o gráfico IMU {save_path}: {e}")
    plt.close()

if __name__ == "__main__":
    # Validação dos diretórios de entrada
    if not os.path.isdir(PRE_FILTERED_PPG_DIR):
        print(f"🚨 ERRO: Diretório de entrada PPG não encontrado: '{os.path.abspath(PRE_FILTERED_PPG_DIR)}'")
        exit()
        
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Diretório de saída para os gráficos: {os.path.abspath(PLOT_OUTPUT_DIR)}")

    # Usa os arquivos PPG como guia principal para o loop
    ppg_files = sorted([f for f in os.listdir(PRE_FILTERED_PPG_DIR) if f.endswith("_ppg.npy")])
    
    print(f"\n🔍 {len(ppg_files)} arquivos de sinal PPG encontrados para plotar.")

    for ppg_filename in ppg_files:
        base_name = ppg_filename.replace('_ppg.npy', '')
        print(f"\n--- Processando registro: {base_name} ---")

        # --- Processa e plota o sinal PPG ---
        try:
            ppg_path = os.path.join(PRE_FILTERED_PPG_DIR, ppg_filename)
            ppg_signal = np.load(ppg_path)
            plot_ppg_signal(ppg_signal, base_name, PLOT_OUTPUT_DIR)
        except Exception as e:
            print(f"  ❌ ERRO ao processar o arquivo PPG {ppg_filename}: {e}")
            continue

        # --- Encontra, processa e plota o sinal IMU correspondente ---
        imu_filename = f"{base_name}_imu.npy"
        imu_path = os.path.join(PRE_FILTERED_IMU_DIR, imu_filename)
        
        if os.path.exists(imu_path):
            try:
                imu_signal = np.load(imu_path)
                plot_imu_signal(imu_signal, base_name, PLOT_OUTPUT_DIR)
            except Exception as e:
                print(f"  ❌ ERRO ao processar o arquivo IMU {imu_filename}: {e}")
        else:
            print(f"  ℹ️  Arquivo IMU correspondente não encontrado para {base_name}.")

    print(f"\n--- Plotagem Concluída ---")