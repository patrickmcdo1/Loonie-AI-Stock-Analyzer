import os
import sys
import pandas as pd
import numpy as np
import random
import argparse
import json
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path

# === Fix Windows encoding issues ===
# Set UTF-8 encoding for stdout/stderr to handle emoji characters
if sys.platform == 'win32':
    try:
        # Python 3.7+
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# === Parse Command Line Arguments ===
parser = argparse.ArgumentParser(description='Run backtest on stock forecast predictions')
parser.add_argument('--forecast-dir', type=str, default='forecasts',
                    help='Directory containing forecast CSV files (default: forecasts)')
parser.add_argument('--model-name', type=str, default='',
                    help='Model name identifier for output files (e.g., old_model, new_model)')
parser.add_argument('--video-dir', type=str, default='videos',
                    help='Directory to save output videos (default: videos)')
parser.add_argument('--output-dir', type=str, default='backtest_results',
                    help='Directory to save metrics JSON/CSV (default: backtest_results)')
args = parser.parse_args()

# === CONFIG ===
video_dir = args.video_dir
forecast_dir = args.forecast_dir
model_name = args.model_name
output_dir = args.output_dir

# Create output directories
os.makedirs(video_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Add model name suffix to filenames if provided
if model_name:
    model_suffix = f"_{model_name}"
else:
    model_suffix = ""

# Parameters
initial_value = 1.0
random_runs = 100
CONFIDENCE_THRESHOLD = 0.70  # Only buy if P(up) > 70%
CONFIDENCE_Z = 1.5           # For subtract_std adjustment
UNCERTAINTY_MODE = "subtract_std"
SPIKE_THRESHOLD = 0.3        # Reject stocks with daily abs change > 50%

# === Load Forecast Data ===
all_forecasts = {}
rejected_tickers = []

for filename in os.listdir(forecast_dir):
    if filename.endswith("_forecast.csv"):
        filepath = os.path.join(forecast_dir, filename)
        ticker = filename.split("_forecast")[0]
        df = pd.read_csv(filepath, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df["Ticker"] = ticker

        # === Spike filter ===
        if "Close" in df.columns:
            df["pct_change"] = df["Close"].pct_change()
            max_change = df["pct_change"].abs().max()

            if max_change > SPIKE_THRESHOLD:
                rejected_tickers.append((ticker, max_change))
                continue  # reject this ticker entirely

            # Compute actual returns
            df["Actual_LogR_1d"] = np.log(df["Close"].shift(-1) / df["Close"])
            df["Actual_LogR_1w"] = np.log(df["Close"].shift(-5) / df["Close"])
            df["Actual_LogR_1m"] = np.log(df["Close"].shift(-21) / df["Close"])
            df["Actual_LogR_6m"] = np.log(df["Close"].shift(-126) / df["Close"])

        all_forecasts[ticker] = df

# === Report rejected tickers ===
if rejected_tickers:
    print("\n[WARNING] Rejected tickers due to unrealistic spikes (>50% daily change):")
    for t, m in rejected_tickers:
        print(f"  - {t}: max daily change = {m:.2%}")
else:
    print("\n[OK] No stocks rejected for excessive daily change.")

# === Common Dates ===
if not all_forecasts:
    raise ValueError("No valid forecast files found after filtering.")
all_dates = [set(df.index) for df in all_forecasts.values()]
common_dates = sorted(set.intersection(*all_dates))
print(f"\n🕒 Using {len(common_dates)} common dates across {len(all_forecasts)} valid stocks")

# === Uncertainty adjustment helper ===
def adjust_prediction(pred, std, mode=UNCERTAINTY_MODE):
    if np.isnan(pred):
        return np.nan
    std = 0.0 if (std is None or np.isnan(std)) else std
    if mode == "subtract_std":
        return pred - CONFIDENCE_Z * std
    elif mode == "sharpe":
        return pred / (1 + std / (abs(pred) + 1e-9))
    elif mode == "expected_positive":
        if std == 0:
            return pred
        p_up = 1 - norm.cdf(0, loc=pred, scale=std)
        return p_up * pred
    return pred

# === Multi-horizon Adaptive Strategy ===
strategy_value = initial_value
strategy_history = []
buy_points = []
successful_buys = 0
total_buys = 0

periods = {"1d": 1, "1w": 5, "1m": 21, "6m": 126}
hold_days_remaining = 0
current_hold = None

for date in common_dates:
    if hold_days_remaining > 0:
        hold_days_remaining -= 1
        strategy_history.append(strategy_value)
        continue

    if current_hold is not None:
        realized = current_hold["Actual_LogR"]
        if not np.isnan(realized):
            gain = np.exp(realized) - 1
            if gain > 0:
                successful_buys += 1
            total_buys += 1
            strategy_value *= np.exp(realized)
        current_hold = None

    candidates = []
    for ticker, df in all_forecasts.items():
        if date not in df.index:
            continue
        row = df.loc[date]

        for label, days in periods.items():
            pred_col = f"Pred_LogR_{label}"
            std_col = f"Pred_LogR_PerDayStd_{label}"
            act_col = f"Actual_LogR_{label}"

            if pred_col not in row or np.isnan(row[pred_col]):
                continue

            pred = float(row[pred_col])
            std = float(row[std_col]) if std_col in row else 0.0
            std = max(std, 1e-4)
            adj_pred = adjust_prediction(pred, std)
            act = float(row[act_col]) if act_col in row else np.nan
            p_up = 1 - norm.cdf(0, loc=pred, scale=std)

            candidates.append({
                "Ticker": ticker,
                "Period": label,
                "Days": days,
                "Predicted": pred,
                "Std": std,
                "AdjPred": adj_pred,
                "Confidence": p_up,
                "Actual_LogR": act,
                "NormAdj": adj_pred / days
            })

    if not candidates:
        strategy_history.append(strategy_value)
        continue

    best = max(candidates, key=lambda x: x["NormAdj"])

    # Only buy if high confidence
    if best["AdjPred"] > 0 and best["Confidence"] >= CONFIDENCE_THRESHOLD:
        current_hold = best
        hold_days_remaining = best["Days"]
        buy_points.append({
            "Date": date,
            "Value": strategy_value,
            "Ticker": best["Ticker"],
            "Confidence": best["Confidence"],
            "Predicted": best["Predicted"]
        })

    strategy_history.append(strategy_value)

# === Report Buy Success Rate ===
if total_buys > 0:
    success_fraction = successful_buys / total_buys
    print(f"\n[STATS] Buy success rate: {successful_buys}/{total_buys} = {success_fraction:.2%}")
else:
    print("\n[WARNING] No completed trades to evaluate.")
    success_fraction = 0.0


# === Random Baseline ===
returns_by_date = {}
for date in common_dates:
    daily_data = []
    for ticker, df in all_forecasts.items():
        if date in df.index and "Actual_LogR_1d" in df.columns:
            val = df.loc[date, "Actual_LogR_1d"]
            if not pd.isna(val):
                daily_data.append(val)
    returns_by_date[date] = daily_data

random_results = np.zeros((len(common_dates), random_runs))
for run in range(random_runs):
    value = initial_value
    for i, date in enumerate(common_dates):
        if returns_by_date[date]:
            pick = random.choice(returns_by_date[date])
            value *= np.exp(pick)
        random_results[i, run] = value

random_mean = np.mean(random_results, axis=1)
random_std = np.std(random_results, axis=1)

# === SPY Buy & Hold ===
spy_path = "TrainingData/indicators_data/processed/SPY-VIX/SPY_daily_processed.csv"
spy_df = pd.read_csv(spy_path, parse_dates=["date"])
spy_df = spy_df.rename(columns={"date": "Date", "close": "Close"})
spy_df = spy_df[spy_df["Date"].isin(common_dates)].sort_values("Date").reset_index(drop=True)
spy_df["PortfolioValue"] = initial_value * (spy_df["Close"] / spy_df["Close"].iloc[0])

# === Calculate Additional Metrics ===
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio from returns."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

def calculate_max_drawdown(values):
    """Calculate maximum drawdown from portfolio values."""
    if len(values) == 0:
        return 0.0
    cumulative = np.array(values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

# Calculate returns from strategy history
strategy_returns = []
for i in range(1, len(strategy_history)):
    if strategy_history[i-1] > 0:
        ret = (strategy_history[i] - strategy_history[i-1]) / strategy_history[i-1]
        strategy_returns.append(ret)

final_value = strategy_history[-1] if strategy_history else initial_value
total_return = (final_value - initial_value) / initial_value
sharpe = calculate_sharpe_ratio(np.array(strategy_returns)) if strategy_returns else 0.0
max_drawdown = calculate_max_drawdown(strategy_history)
win_rate = success_fraction if total_buys > 0 else 0.0

# SPY metrics for comparison
spy_final = spy_df["PortfolioValue"].values[-1] if len(spy_df) > 0 else initial_value
spy_total_return = (spy_final - initial_value) / initial_value

print(f"\n[STRATEGY METRICS]")
print(f"   Final Value: {final_value:.4f}")
print(f"   Total Return: {total_return:.2%}")
print(f"   Sharpe Ratio: {sharpe:.4f}")
print(f"   Max Drawdown: {max_drawdown:.2%}")
print(f"   Win Rate: {win_rate:.2%}")
print(f"   Total Trades: {total_buys}")
print(f"\n[SPY BUY & HOLD]")
print(f"   Final Value: {spy_final:.4f}")
print(f"   Total Return: {spy_total_return:.2%}")

# === Save Metrics to JSON ===
metrics = {
    "model_name": model_name if model_name else "default",
    "forecast_dir": forecast_dir,
    "initial_value": initial_value,
    "final_value": float(final_value),
    "total_return": float(total_return),
    "sharpe_ratio": float(sharpe),
    "max_drawdown": float(max_drawdown),
    "win_rate": float(win_rate),
    "successful_buys": int(successful_buys),
    "total_buys": int(total_buys),
    "total_dates": len(common_dates),
    "num_stocks": len(all_forecasts),
    "rejected_tickers": len(rejected_tickers),
    "spy_final_value": float(spy_final),
    "spy_total_return": float(spy_total_return),
    "beats_spy": bool(total_return > spy_total_return),
    "return_over_spy": float(total_return - spy_total_return)
}

metrics_file = os.path.join(output_dir, f"metrics{model_suffix}.json")
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n[SAVED] Metrics saved to: {metrics_file}")

# === Animation ===
def animate_plot(dates, random_results, strat_values, spy_values,
                 random_mean=None, random_std=None, show_uncertainty=False, filename="out.mp4"):
    dates = np.asarray(dates)
    step = max(1, len(dates) // 500)
    frames = range(0, len(dates), step)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    random_lines = [ax.plot([], [], color="white", alpha=0.10, lw=1)[0] for _ in range(random_results.shape[1])]
    spy_line, = ax.plot([], [], color="white", lw=3, label="SPY Buy & Hold")
    strat_line, = ax.plot([], [], color="#39FF14", lw=3, label="AI Strategy")

    if show_uncertainty and random_mean is not None and random_std is not None:
        ax.fill_between(dates, random_mean - 3*random_std, random_mean + 3*random_std,
                        color="gray", alpha=0.08, label="±3σ Random Range")
        ax.fill_between(dates, random_mean - random_std, random_mean + random_std,
                        color="gray", alpha=0.2, label="±1σ Random Range")

    ymin, ymax = np.nanmin(random_results), np.nanmax(np.concatenate([random_results.flatten(), strat_values, spy_values]))
    margin = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_title("AI Strategy vs Random vs SPY", color="white", fontsize=22)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Portfolio Value", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black", edgecolor="white", fontsize=12)
    for text in ax.get_legend().get_texts():
        text.set_color("white")

    def update(i):
        for r, line in enumerate(random_lines):
            line.set_data(dates[:i], random_results[:i, r])
        spy_line.set_data(dates[:i], spy_values[:i])
        strat_line.set_data(dates[:i], strat_values[:i])
        return [spy_line, strat_line]
        

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    out_path = os.path.join(video_dir, filename)
    writer = FFMpegWriter(fps=60, bitrate=1800)
    anim.save(out_path, writer=writer, dpi=220)
    print(f"[SAVED] Animation saved: {out_path}")
    plt.close(fig)

# === Run animations ===
animate_plot(common_dates, random_results, strategy_history, spy_df["PortfolioValue"].values,
             show_uncertainty=False, filename=f"random_vs_strategy_clean{model_suffix}.mp4")

animate_plot(common_dates, random_results, strategy_history, spy_df["PortfolioValue"].values,
             random_mean, random_std, show_uncertainty=True, filename=f"random_vs_strategy_uncertainty{model_suffix}.mp4")
