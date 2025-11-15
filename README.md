# AI Stock Predictor

A deep learning pipeline for forecasting short-term stock market movements using price indicators, Monte Carlo dropout uncertainty estimation, and walk-forward validation.

# Overview

This project builds an end-to-end machine learning pipeline that:

- Loads and processes stock indicator data from CSV files
- Generates normalized training windows
- Trains a Conv1D + LSTM model with Monte Carlo dropout
- Uses uncertainty to determine confidence in predictions
- Outputs buy/hold forecasts to the /forecasts folder

The design emphasizes realistic backtesting, uncertainty-aware predictions, and ease of extension — ideal for research, education, or algorithmic-trading experimentation.
```
├── forecasts/                  # Original model forecast CSV outputs
├── forecasts_improved/         # Improved model forecast CSV outputs
├── cache/                      # Cached preprocessed numpy arrays (original)
├── cache_improved/             # Cached preprocessed numpy arrays (improved)
├── models_improved/            # Saved model checkpoints (improved model)
├── backtest_results/           # Backtest metrics and comparison results
├── videos/                     # Rendered content for visualization or YouTube
├── TrainingData/
│   ├── indicators_data/
│   │   ├── raw/                # Raw scraped data (price, sentiment, insider)
│   │   └── processed/
│   │       ├── SPY-VIX/        # Market indicators
│   │       └── stocksData/     # Stock CSVs (one per ticker)
│   ├── featuresPy/             # Feature-generation scripts
│   ├── processor.py            # Main feature pipeline
│   └── downloader.py           # Data download helpers
├── forecast.ipynb              # Original Jupyter notebook for running forecasts
├── forecast_improved.ipynb     # Enhanced notebook with bug fixes and improvements
├── forecasting_backtest_Predictor.py  # Original backtest script
├── forecasting_backtest_Predictor_v2.py # Enhanced backtest with CLI args and metrics
├── compare_models.py           # Tool to compare old vs new model performance
├── config.json                 # API key configuration (create this file)
└── README.md
```

## **Available Features (Indicators)**

`close, YesterdayClose, YesterdayOpenLogR, YesterdayHighLogR, YesterdayLowLogR, YesterdayVolumeLogR, YesterdayCloseLogR, MA10, MA20, MA30, DayOfWeek, DayOfMonth, MonthNumber, EMA10, EMA30, RSI, MACD, MACD_Signal, BollingerUpper, BollingerLower, Volatility_10, Volatility_20, Volatility_30, OBV, ZScore, insider_shares, insider_amount, insider_buy_flag, sentiment, num_articles, overnight_gap, abnormal_vol, volatility_5d, volatility_20d, momentum_5d, momentum_20d, skew_5d, intraday_range, sentiment_change`

Each is automatically merged, cleaned, and normalized during preprocessing.

---

Key Features
FeatureDescriptionConv1D + LSTM ArchitectureLearns short- and long-term dependencies in stock dataMonte Carlo DropoutProduces uncertainty estimates for each forecastWalk-Forward ValidationPrevents data leakage, simulates real-time tradingRegime and Volatility AwarenessDetects market conditions for more robust signalsBatch Data GeneratorLoads multiple stock datasets efficiently from cacheForecast Confidence ThresholdTrades only when confidence > threshold (default 0.7)

## Key Features

| Feature | Description |
|----------|-------------|
| **Conv1D + LSTM Architecture** | Learns short- and long-term dependencies in stock data |
| **Attention Mechanism** (Improved) | Weights important time steps for better pattern recognition |
| **Monte Carlo Dropout** | Produces uncertainty estimates for each forecast |
| **Walk-Forward Validation** | Prevents data leakage, simulates real-time trading |
| **Comprehensive Evaluation** (Improved) | Test set metrics, baseline comparisons, risk-adjusted returns |
| **Batch Data Generator** | Loads multiple stock datasets efficiently from cache |
| **Forecast Confidence Threshold** | Trades only when confidence > threshold (default 0.7) |
| **GPU Acceleration** (Improved) | Configurable GPU support for multi-GPU systems |
| **Model Comparison Tools** | Automated backtest comparison between model versions |

## Getting Started

### 1. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```
### 2. Training & Generating Predictions

There are two notebooks available for training:

#### `forecast.ipynb` (Original Model)
The original notebook that trains the baseline model:
- Basic Conv1D + LSTM architecture
- Monte Carlo dropout for uncertainty estimation
- Outputs forecasts to `forecasts/` directory
- Uses GPU:0 by default

#### `forecast_improved.ipynb` (Enhanced Model) ⭐ **Recommended**
The improved notebook with critical bug fixes and enhancements:
- **Bug Fixes**: Fixed column overwriting bug, data leakage in scaler, feature importance issues
- **Architecture**: Enhanced with attention mechanism, dual Conv1D + dual LSTM layers
- **Evaluation**: Comprehensive test set metrics (Sharpe ratio, directional accuracy, etc.)
- **Data Handling**: Better missing data handling, outlier clipping, optional PCA
- **GPU Support**: Configured to use GPU:1 (configurable for multi-GPU systems)
- **Outputs**: Forecasts saved to `forecasts_improved/` directory

**To train:**
1. Open either `forecast.ipynb` or `forecast_improved.ipynb` in Jupyter
2. Run all cells to train the model
3. The notebook will automatically use processed stocks from `/TrainingData/indicators_data/processed/stocksData/`
4. There is a sample of limited stocks already included in the package, however, more data could be added

**GPU Configuration:**
- The improved notebook (`forecast_improved.ipynb`) is configured to use GPU:1 by default
- This allows parallel execution alongside the original notebook (which uses GPU:0)
- GPU configuration can be modified in the first cell of the notebook 

### 3. Creating Historical Datasets (Optional)

If you want to generate your own datasets from scratch:

#### Step 1: Configure AlphaVantage API Key

The downloader scripts use **AlphaVantage** as the primary data source for historical stock prices and market data. To access this data, you'll need a free API key:

1. Sign up for a free API key at [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Create a `config.json` file in the project root directory with the following structure:
   ```json
   {
     "ALPHA_VANTAGE_KEY": "your_api_key_here"
   }
   ```
3. Replace `"your_api_key_here"` with your actual AlphaVantage API key
4. Save the file

**Note:** 
- The API key is stored in `config.json` (not hardcoded in scripts) for security
- Make sure `config.json` is in `.gitignore` to avoid committing your API key
- AlphaVantage's free tier has rate limits (typically 5 API calls per minute and 500 calls per day). The downloader scripts include built-in rate limiting to respect these constraints.
- SEC EDGAR requires a User-Agent with contact information (see `TrainingData/featuresPy/insiderbuying.py` for configuration)

#### Step 2: Run `TrainingData/downloader.py`

```bash
python TrainingData/downloader.py
```

This script will:
- Download historical price data from AlphaVantage
- Fetch sentiment data and insider trading information from SEC EDGAR and other sources
- Store raw data in `/TrainingData/indicators_data/raw/`

#### Step 3: Run `TrainingData/processor.py`

```bash
python TrainingData/processor.py
```

This script will:
- Clean and merge raw data
- Compute all technical indicators
- Merge insider trading and sentiment data
- Output final processed CSVs to: `/TrainingData/indicators_data/processed/stocksData/`

---

## Model Comparison & Backtesting

### Comparing Old vs New Model

After training both models, use the comparison tool to evaluate their performance:

```bash
python compare_models.py
```

This script will:
1. Run backtests on both `forecasts/` (old model) and `forecasts_improved/` (new model)
2. Generate comprehensive performance metrics for each
3. Create side-by-side comparison report
4. Save results to `backtest_results/`:
   - `metrics_old_model.json` - Original model metrics
   - `metrics_new_model.json` - Improved model metrics
   - `model_comparison.csv` - Side-by-side comparison table
   - `model_comparison.json` - Detailed JSON comparison

### Running Individual Backtests

You can also run backtests on individual models using `forecasting_backtest_Predictor_v2.py`:

```bash
# Backtest original model
python forecasting_backtest_Predictor_v2.py --forecast-dir forecasts --model-name old_model

# Backtest improved model
python forecasting_backtest_Predictor_v2.py --forecast-dir forecasts_improved --model-name new_model
```

**Available CLI Arguments:**
- `--forecast-dir`: Directory containing forecast CSV files (default: `forecasts`)
- `--model-name`: Model identifier for output files (e.g., `old_model`, `new_model`)
- `--video-dir`: Directory to save output videos (default: `videos`)
- `--output-dir`: Directory to save metrics JSON (default: `backtest_results`)

**Backtest Outputs:**
- Strategy performance videos (with/without uncertainty bands)
- Detailed metrics JSON (total return, Sharpe ratio, win rate, etc.)
- Comparison against SPY buy-and-hold and random baseline

### Enhanced Evaluation Features

The improved model (`forecast_improved.ipynb`) includes comprehensive evaluation:

**Test Set Metrics:**
- MSE, MAE, RMSE for each prediction horizon
- Directional accuracy (up/down classification)
- Information Coefficient (prediction-actual correlation)
- Win rate and profit factor
- Uncertainty calibration metrics

**Baseline Comparisons:**
- Random Walk baseline
- Zero prediction (mean reversion)
- Moving Average baseline
- SPY buy-and-hold comparison

**Risk Metrics:**
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside volatility only)
- Maximum Drawdown
- Win rate and trade statistics

All metrics are automatically saved to `forecasts_improved/test_metrics.csv` after training.

---

![Backtest Example](images/image1.PNG)