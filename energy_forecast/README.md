# 🛢️ Energy Commodity Price Forecaster

Projecto Python para previsão de preços de commodities de energia com modelos **ARIMA/SARIMA**.

---

## 📁 Estrutura do Projecto

```
energy_forecast/
├── main.py                  ← Pipeline principal (ponto de entrada)
├── requirements.txt         ← Dependências
├── data/
│   ├── fetcher.py           ← Download e cache de dados via yfinance
│   └── commodities.csv      ← Cache gerada automaticamente
├── models/
│   ├── arima_model.py       ← Classe ArimaForecaster + testes de estacionaridade
│   └── plots.py             ← Todas as funções de visualização
└── results/                 ← Gráficos e CSVs gerados automaticamente
    ├── all_series.png
    ├── correlation.png
    ├── <commodity>_acf_pacf.png
    ├── <commodity>_stationarity.png
    ├── <commodity>_diagnostics.png
    ├── <commodity>_validation.png
    ├── <commodity>_forecast.png
    ├── <commodity>_forecast.csv
    └── summary.csv
```

---

## 🚀 Instalação

```bash
# 1. Clonar / copiar o projecto
cd energy_forecast

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependências
pip install -r requirements.txt
```

---

## ▶️ Utilização

### Execução básica (30 dias de previsão)
```bash
python main.py
```

### Opções disponíveis
```bash
python main.py --forecast-days 60   # prever 60 dias úteis
python main.py --test-size 90       # 90 observações no teste
python main.py --seasonal           # usar SARIMA (componente sazonal semanal)

# Combinar opções
python main.py --forecast-days 30 --test-size 60 --seasonal
```

---

## 📊 Commodities Cobertas

| Nome         | Ticker  | Descrição                          |
|--------------|---------|------------------------------------|
| WTI_Oil      | CL=F    | Petróleo WTI (West Texas)          |
| Brent_Oil    | BZ=F    | Petróleo Brent (referência europeia)|
| Natural_Gas  | NG=F    | Gás Natural (Henry Hub)            |
| Coal         | KOL     | Carvão (ETF VanEck Coal)           |
| Electricity  | ICLN    | Eletricidade (ETF iShares Clean Energy) |

---

## 🔬 Pipeline Detalhado

### 1. Download de Dados
- Fonte: **Yahoo Finance** via `yfinance`
- Histórico desde 2015 (≈ 10 anos)
- Cache automática em `data/commodities.csv` (evita downloads repetidos)

### 2. Análise Exploratória
- Gráfico de séries temporais individuais
- Matriz de correlação dos retornos diários

### 3. Por Commodity
1. **ACF / PACF** – identificação visual de autocorrelações
2. **Testes de estacionaridade**
   - Augmented Dickey-Fuller (ADF): H₀ = raiz unitária → *p < 0.05* ⟹ estacionária
   - KPSS: H₀ = estacionária → *p > 0.05* ⟹ estacionária
3. **Auto-ARIMA** – `pmdarima.auto_arima` selecciona (p,d,q) por AIC
4. **Treino SARIMAX** – via `statsmodels`
5. **Diagnóstico** – análise de resíduos (4 gráficos)
6. **Validação Walk-Forward** – previsão 1-passo com expansão de janela
   - Métricas: MAE, RMSE, MAPE
7. **Previsão** – horizonte futuro com intervalo de confiança 95%

### 4. Resumo
- `results/summary.csv` com métricas e previsão final de cada commodity

---

## 📈 Exemplo de Output (summary.csv)

| Commodity   | Modelo       | MAE    | RMSE   | MAPE_% | Previsão_30d |
|-------------|--------------|--------|--------|--------|--------------|
| WTI_Oil     | ARIMA(1,1,1) | 1.23   | 1.87   | 1.54   | 78.32        |
| Natural_Gas | ARIMA(2,1,2) | 0.08   | 0.11   | 2.31   | 2.64         |
| ...         | ...          | ...    | ...    | ...    | ...          |

---

## 🧩 Extensões Sugeridas

| Ideia                     | Como implementar                                      |
|---------------------------|-------------------------------------------------------|
| Mais features (exógenas)  | Usar SARIMAX com variáveis como USD index, temperatura|
| Comparar com LSTM         | Adicionar `models/lstm_model.py`                      |
| Dashboard interactivo     | Integrar com `streamlit` ou `dash`                    |
| Alertas de preço          | Adicionar notificações por email / Telegram           |
| Deploy automatizado       | Agendar com `cron` ou GitHub Actions                  |

---

## 📝 Notas

- Os dados são preços de fecho de **futuros / ETFs** e **não** constituem aconselhamento financeiro.
- O ARIMA funciona melhor em horizontes **curtos** (≤ 30 dias). Para horizontes longos considera modelos híbridos.
- A primeira execução pode demorar 2–5 minutos (download + auto_arima).
