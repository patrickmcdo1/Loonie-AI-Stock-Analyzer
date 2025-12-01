import yfinance as yf
import pandas as pd
import os

def download_market_data(save_folder="TrainingData/indicators_data/raw/stocksData", period="10y"):
    """
    Download market context data (SPY for market, VIX for volatility)
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    market_symbols = {
        'SPY': 'S&P 500 ETF',
        '^VIX': 'VIX Volatility Index'
    }

    for symbol, name in market_symbols.items():
        try:
            print(f"⬇️  Downloading full daily data for {symbol} ({name})...")
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                print(f"❌ No data found for {symbol}")
                continue
            
            # Reset index and standardize columns
            data.reset_index(inplace=True)
            data.columns = [col.lower() for col in data.columns]
            
            # Keep OHLCV data
            columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
            data = data[[col for col in columns_to_keep if col in data.columns]]
            
            # Sort and save
            data.sort_values('date', inplace=True)
            data.reset_index(drop=True, inplace=True)
            
            # Use clean filename (replace ^ with nothing)
            clean_symbol = symbol.replace('^', '')
            filepath = os.path.join(save_folder, f"{clean_symbol}_daily.csv")
            data.to_csv(filepath, index=False)
            
            print(f"✅ Saved {symbol} data ({len(data)} rows) to {filepath}\n")
            
        except Exception as e:
            print(f"❌ Failed to download {symbol}: {e}\n")

if __name__ == "__main__":
    print("🌍 Downloading market data...\n")
    print("=" * 60 + "\n")
    download_market_data(period="10y")
    print("✅ Market data download complete!")


