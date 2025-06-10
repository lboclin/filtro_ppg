Análise de Sinais PPG para Estimativa de Frequência Cardíaca na Presença de Artefatos de Movimento
Descrição do Projeto
Este projeto implementa um pipeline completo de processamento de sinais em Python para estimar a frequência cardíaca (BPM) a partir de dados de fotopletismografia (PPG). O principal desafio abordado é a supressão de artefatos de movimento (AM), que são a maior fonte de erro em medições de PPG feitas em sujeitos não estacionários.

O algoritmo final utiliza uma abordagem híbrida que analisa o sinal PPG em conjunto com dados de um acelerômetro (IMU) para validar a qualidade do sinal e tomar decisões inteligentes, descartando medições não confiáveis e resolvendo ambiguidades entre o ritmo do coração e o ritmo do movimento.

Estrutura do Projeto
O repositório está organizado da seguinte forma:

filtro_ppg/
│
├── data/               # (IGNORADO PELO GIT) Contém os dados brutos, intermediários e finais.
├── results/              # Contém os resultados finais em formato .csv.
│   ├── ground_truth/     # BPM calculado a partir do ECG (padrão-ouro).
│   └── bpm_vfinal/       # BPM calculado pelo nosso algoritmo final.
├── outputs/              # Contém todas as visualizações e gráficos gerados.
│   ├── output_pre_filtered/
│   ├── output_filtered_1/
│   └── final_comparison/
└── scripts/              # Contém todo o código-fonte do pipeline.
O Pipeline de Processamento
O projeto é executado através de uma sequência de scripts, cada um responsável por uma etapa do processo:

1. Extração e Preparação (salvar_raw_para_npy.py)

Lê os dados brutos do formato WFDB.
Extrai os canais de interesse (PPG e IMU de 3 eixos).
No caso de múltiplos canais PPG, calcula a média para gerar um único sinal com melhor relação sinal-ruído.
Salva os sinais "crus" em formato .npy para facilitar o manuseio.
2. Pré-processamento e Filtragem (passa_faixa_ppg.py, passa_baixa_imu.py)

PPG: Aplica-se um filtro passa-faixa para remover flutuações de linha de base e ruídos de alta frequência, isolando a faixa fisiológica do coração.
IMU: Aplica-se um filtro passa-baixa para suavizar o "jitter" eletrônico do sensor, mantendo o sinal de movimento real.
3. Geração do Padrão-Ouro (generate_ground_truth.py)

Para validar nosso algoritmo, este script processa o sinal de ECG (padrão-ouro) usando a biblioteca biosppy.
Ele detecta os picos R com alta precisão e calcula uma série temporal de BPM "verdadeiro", que serve como nosso gabarito.
4. Estimativa de BPM (calculate_bpm_vfinal.py)

Este é o coração do projeto. Ele implementa o algoritmo final que:
Analisa os sinais PPG e IMU em janelas deslizantes de 8 segundos.
Usa a FFT (Transformada Rápida de Fourier) para encontrar a frequência dominante (e sua potência) em ambos os sinais.
Implementa um "detector de mentiras" que compara as duas frequências. Se elas colidem, um critério de desempate baseado na potência do sinal é usado para decidir se o pico do PPG é um batimento cardíaco real ou um artefato de movimento.
5. Análise e Visualização (analise_csv.py, plot_final_comparison.py)

Scripts utilitários para gerar estatísticas (média, mediana) dos resultados e criar gráficos comparativos entre o BPM estimado pelo nosso algoritmo e o BPM real do ECG.
Resultados e o Desafio dos Artefatos de Movimento
Ao analisar os resultados, observamos um desempenho excelente nos cenários de repouso (sit) e caminhada (walk). O algoritmo foi capaz de usar o IMU para corrigir o problema de "Cadence Lock", onde o ritmo dos passos era confundido com o do coração.

No entanto, como pode ser visto nos próprios resultados, o cenário de corrida (run) apresenta o maior desafio. Frequentemente, a frequência cardíaca de uma corrida (~130 BPM) e a cadência dos passos (~130 SPM) são quase idênticas.

Esta é a essência do problema. Nosso algoritmo é projetado para ser conservador. Ele detecta essa "colisão de frequências" e, em muitos casos, quando a energia do movimento é avassaladora, ele corretamente identifica que a medição não é confiável e a descarta. O resultado disso é que, para algumas corridas, o BPM calculado pode parecer baixo ou inconsistente.

Isso não é necessariamente uma falha, mas sim uma característica de um sistema robusto: ele demonstra a capacidade de reconhecer quando a qualidade do sinal de entrada é muito baixa para produzir uma estimativa confiável. A solução para este desafio extremo está na fronteira da pesquisa e aponta para trabalhos futuros.

Trabalhos Futuros
Pós-processamento: Aplicar filtros de suavização (ex: mediana móvel) sobre os resultados de BPM finais para gerar uma curva de tendência mais estável.
Machine Learning: Utilizar os dados preparados para treinar um modelo de rede neural (ex: LSTM) que possa aprender a relação não-linear entre os artefatos de movimento e o sinal PPG, buscando uma solução ainda mais precisa para os cenários de alta intensidade.