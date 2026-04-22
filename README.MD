# 📊 Bitcoin Market Sentiment vs Trader Performance Analysis

> **Hyperliquid DEX Trading Data × Fear & Greed Index**  
> Uncovering hidden patterns to drive smarter trading strategies

---

## 📁 Project Structure

```
PRIME_TRADE.AI_TASK/
│
├── main.ipynb                    # Main analysis notebook
├── historical_data.csv           # Hyperliquid trader data (211,224 rows)
├── fear_greed_index.csv          # Bitcoin Fear & Greed Index (2,644 days)
│
├── historical_data_profile.html  # YData profiling — historical data
├── fear_greed_data_profile.html  # YData profiling — fear/greed data
│
├── README.md                     # This file
│
└── plots/
    ├── pnl_by_sentiment.png          # PnL distribution by sentiment
    ├── win_rate_by_sentiment.png     # Win rate across sentiments
    ├── long_vs_short_sentiment.png   # BUY vs SELL by sentiment
    ├── direction_sentiment.png       # Trade direction analysis
    ├── heatmap_coin_sentiment.png    # Coin × Sentiment heatmap
    ├── daily_pnl_trend.png           # Daily PnL time series
    ├── elbow_silhouette.png          # K-Means optimal K selection
    ├── trader_clusters_pca.png       # Trader cluster visualization
    ├── cluster_sentiment_heatmap.png # Cluster performance by sentiment
    ├── pca_analysis.png              # PCA feature importance
    └── anomaly_detection.png         # Isolation Forest anomalies
```

---

## 🎯 Objective

Explore the relationship between **Bitcoin market sentiment** (Fear & Greed Index)
and **trader performance** on Hyperliquid DEX. The goal is to:

- Uncover hidden patterns between sentiment and profitability
- Identify which sentiment periods produce best/worst returns
- Classify traders into behavioural groups using unsupervised ML
- Detect anomalous trades using Isolation Forest
- Deliver actionable trading strategy recommendations

---

## 📦 Datasets

### 1. Historical Trader Data (Hyperliquid DEX)
| Property | Value |
|----------|-------|
| Rows | 211,224 |
| Columns | 16 |
| Unique Traders | 32 accounts |
| Unique Coins | 246 |
| Date Range | Jan 2024 – May 2025 |
| Missing Values | 0 ✅ |

**Key Columns:**
- `Account` — Trader wallet address
- `Coin` — Trading pair (BTC, ETH, SOL, HYPE, etc.)
- `Execution Price` — Trade execution price
- `Size USD` — Trade size in USD
- `Side` — BUY or SELL
- `Direction` — Open Long / Close Long / Open Short / Close Short
- `Closed PnL` — Profit or Loss on closed trades
- `Timestamp IST` — Trade timestamp (India Standard Time)

### 2. Fear & Greed Index
| Property | Value |
|----------|-------|
| Rows | 2,644 |
| Columns | 4 |
| Date Range | Feb 2018 – May 2025 |
| Missing Values | 0 ✅ |

**Sentiment Categories:**
| Classification | Days |
|----------------|------|
| Fear | 781 |
| Greed | 633 |
| Extreme Fear | 508 |
| Neutral | 396 |
| Extreme Greed | 326 |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn ydata-profiling
```

### Run the Notebook
```bash
jupyter notebook main.ipynb
```

> ⚠️ Make sure `historical_data.csv` and `fear_greed_index.csv`
> are in the **same folder** as `main.ipynb` before running.

---

## 📊 Analysis Pipeline

```
Step 1  →  Load Data
Step 2  →  Clean Column Names & Fix @ Prefix in Coin Column
Step 3  →  Parse Dates & Merge Datasets on Date
Step 4  →  Filter: Keep Only Closed Trades (PnL ≠ 0)
Step 5  →  Feature Engineering (is_winner, ordered sentiment)
Step 6  →  EDA: PnL Distribution by Sentiment
Step 7  →  EDA: Win Rate by Sentiment
Step 8  →  EDA: BUY vs SELL Performance by Sentiment
Step 9  →  EDA: Trade Direction Analysis
Step 10 →  EDA: Coin × Sentiment Heatmap
Step 11 →  EDA: Daily PnL Time Series
Step 12 →  Statistical Testing (T-test + ANOVA)
Step 13 →  ML: K-Means Clustering (Trader Segmentation)
Step 14 →  ML: PCA (Feature Importance)
Step 15 →  ML: Isolation Forest (Anomaly Detection)
Step 16 →  Final Insights & Strategy Recommendations
```

---

## 🔑 Key Findings

### 📈 EDA Findings

| Sentiment | Avg PnL/Trade | Win Rate | Best Side |
|-----------|:-------------:|:--------:|:---------:|
| Extreme Fear | $5 | 82.7% | SELL only |
| Fear | $286 | 85.2% | BUY ($558) |
| Neutral | $45 | 81.6% | SELL slightly |
| Greed | $105 | 84.8% | SELL ($141) |
| **Extreme Greed** | **$339** | **91.0%** | **SELL ($411)** |

### 🏆 Top Discoveries

1. **Extreme Greed = Best Trading Environment**
   - Highest win rate: **91%**
   - Highest avg PnL: **$339/trade**
   - SELL (short) positions earn **17.5× more** than BUY

2. **The Fear Paradox**
   - Fear win rate (85.2%) ≈ Greed win rate (84.8%)
   - BUY during Fear earns **$558 avg** — best BUY condition
   - SOL during Fear = **$1,679 avg PnL** ← best single combo

3. **Contrarian Strategy Works**
   - SELL dominates during Greed ($141 vs $41 BUY)
   - SELL dominates during Extreme Greed ($411 vs $23 BUY)
   - Data proves: *"Be fearful when others are greedy"*

4. **Avoid ETH in Greed Markets**
   - ETH during Greed: **-$243 avg loss**
   - ETH during Extreme Greed: **-$546 avg loss**

5. **Extreme Fear = Stay Out**
   - Avg PnL nearly zero ($5)
   - BUY positions lose money (-$35 avg)

---

### 🤖 Machine Learning Findings

#### K-Means Clustering (K=5, Silhouette-optimised)

| Cluster | Type | Behaviour |
|---------|------|-----------|
| High Risk Traders | Large position size | Great in Fear/Neutral, **CRASHES in Extreme Greed (-$482)** |
| Elite Traders | High PnL + High WR | Consistent across all sentiments ($29–$69) |
| Balanced Traders | Medium risk/reward | Peaks during Fear ($207), contrarian style |
| 🐋 Whale Traders | Outlier accounts | $2M+ total PnL, concentrated in Extreme Greed |

#### PCA Analysis
- **PC1 (33.4% variance)** driven by: `pnl_std` and `avg_pnl`
- **PC2 (23.2% variance)** driven by: `win_rate` and `total_trades`
- **4 components** explain **86%** of all trader variance
- Profit **magnitude** separates traders more than win rate

#### Anomaly Detection (Isolation Forest, 5% contamination)
- **~5,220 trades** flagged as anomalous
- **Fear has highest anomaly rate (6.4%)** → whale accumulation
- **Extreme Fear has lowest anomaly rate (2.2%)** → everyone plays safe
- Anomalous trades have **10× larger** PnL than normal trades

---

## 💡 Trading Strategy Recommendations

| Sentiment | Action | Best Coins | Avoid |
|-----------|--------|------------|-------|
| **Extreme Greed** | SELL (Short) aggressively | 107, BTC, HYPE | ETH longs |
| **Greed** | SELL (Short) | MELANIA, HYPE, SOL | ETH |
| **Fear** | BUY (Long) selectively | SOL, BTC | FARTCOIN |
| **Extreme Fear** | Reduce / Avoid | BTC only if forced | Everything else |
| **Neutral** | Small SELL | PURR/USDC, BTC | — |

---

## 📉 Statistical Tests

| Test | Groups | p-value | Result |
|------|--------|---------|--------|
| T-test | Fear vs Greed | < 0.05 | ✅ Significant |
| T-test | Extreme Fear vs Extreme Greed | < 0.05 | ✅ Significant |
| One-way ANOVA | All 5 groups | < 0.05 | ✅ Significant |

> Sentiment has a **statistically significant** impact on trader PnL.

---

## 👤 Author

**Data Science Candidate** -- Satyabrata Das Adhikary 
Assignment for: **Prime Trade AI**  
Date: **April 2026**

---

## 📄 License

This project is submitted as part of a hiring assignment for Prime Trade AI.
Data is proprietary and used solely for analytical purposes.
