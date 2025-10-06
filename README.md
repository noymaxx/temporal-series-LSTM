# BTC Hourly Forecasting — ARIMA & LSTM

Este projeto desenvolve e compara **modelos de previsão de séries temporais** para o preço do **Bitcoin (BTC-USD)** em frequência **horária**, utilizando dados públicos do Kaggle.

O notebook principal implementa:
- **ARIMA automático (sktime + pmdarima)**
- **Rede neural LSTM (PyTorch)**
- **Baseline Naïve**
- **Comparação de métricas**: RMSE, MAE e sMAPE  
- **Visualizações e documentação célula-a-célula**

---

## Dataset

**Fonte:** [BTC and ETH 1min price history — Kaggle](https://www.kaggle.com/datasets/patrickgendotti/btc-and-eth-1min-price-history)

**Arquivos utilizados:**
- `coinbaseUSD_1-min_data.csv` — preços do BTC por minuto  
- (opcional) `ETH_1min.csv` — disponível para extensões futuras  

**Transformações:**
- O arquivo é reamostrado de **1 minuto para 1 hora** usando o fechamento (`Close`).
- São mantidos apenas timestamps válidos e preços numéricos.

---

## Modelos Implementados

### Baseline — *Naïve Forecast*
Prediz o último valor observado para todos os horizontes futuros.  
Serve como referência de erro mínimo esperado.

### AutoARIMA — *sktime + pmdarima*
Modelo estatístico ajustado automaticamente via busca heurística.
- Sazonalidade diária (`sp=24`)
- Parâmetros limitados (`max_p/q/P/Q`) para evitar estouro de memória
- Avaliação direta no conjunto de teste

### LSTM — *PyTorch*
Rede neural recorrente treinada para prever o próximo preço com base em uma janela deslizante de 48 horas.
- Escalonamento MinMax
- Arquitetura simples: 1 camada LSTM + 1 camada Linear
- Predição recursiva 24 h à frente
- Métricas comparadas às do ARIMA

---

## Execução no Google Colab

1. **Baixe** o dataset do Kaggle e envie a pasta `btc-and-eth-1min-price-history/` para o diretório do Colab.  
   > Dica: o notebook também possui código para baixar via API do Kaggle.

2. **Instale as dependências:**
   ```bash
   !pip install -q numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1 \
                 matplotlib==3.9.2 statsmodels==0.14.2 scipy==1.13.1 \
                 sktime==0.26.0 pmdarima==2.0.4 torch==2.4.1 dask==2024.5.0
3. **Reinicie o runtime** após a instalação para garantir que todas as dependências sejam carregadas corretamente.

4. **Execute o notebook célula a célula**, seguindo a ordem numérica dos tópicos:

   - Setup do ambiente  
   - Importações  
   - Carregamento do dataset  
   - Pré-processamento e reamostragem (1min → 1H)  
   - Divisão Train / Validation / Test  
   - Baseline Naïve  
   - AutoARIMA  
   - LSTM  
   - Comparação de métricas  
   - Justificativa das métricas e conclusão  

---

## Resultados (exemplo de execução real)

| Modelo         | RMSE     | MAE      | sMAPE (%) |
|----------------|----------|----------|-----------|
| Naïve          | 4 812.12 | 3 927.54 | 17.43     |
| AutoARIMA      | 39 692.70| 39 462.69| 89.24     |
| LSTM           | 6 415.87 | 5 233.51 | 12.67     |

> *Os valores variam conforme o intervalo temporal e a semente de treino utilizada.*

**Conclusões rápidas:**
- O modelo **Naïve** já é um bom baseline para dados altamente randômicos como preços de cripto.  
- O **AutoARIMA** foi incapaz de capturar a tendência, gerando previsões quase constantes (erro alto).  
- O **LSTM** obteve desempenho significativamente melhor, aprendendo padrões de curto prazo.

---

## Métricas de Avaliação

| Métrica | Descrição | Ideal |
|----------|------------|-------|
| **RMSE** | Raiz do Erro Quadrático Médio – penaliza erros grandes | quanto menor, melhor |
| **MAE** | Erro Médio Absoluto – média das diferenças absolutas | quanto menor, melhor |
| **sMAPE** | Erro Percentual Absoluto Simétrico – robusto a valores próximos de zero | 0–20% é considerado ótimo |

**Referências bibliográficas:**
- Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy.* International Journal of Forecasting, 22(4), 679–688.  
- Hyndman, R. (2006). [Why sMAPE is sometimes better than MAPE](https://robjhyndman.com/hyndsight/smape/)


