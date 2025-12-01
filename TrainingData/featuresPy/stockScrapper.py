import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def download_yfinance_daily(tickers, save_folder="TrainingData/indicators_data/raw/stocksData", period="10y"):
    """
    Download stock data using yfinance (free, no API key needed)
    
    Args:
        tickers: List of stock symbols
        save_folder: Where to save CSV files
        period: Time period for historical data (default 10y for 10 years)
                Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    total = len(tickers)
    start_time = time.time()

    for i, ticker in enumerate(tickers, 1):
        try:
            filepath = os.path.join(save_folder, f"{ticker}_daily.csv")
            file_exists = os.path.exists(filepath)
            should_download = True

            # Check if file exists and is up to date
            if file_exists:
                existing_data = pd.read_csv(filepath, parse_dates=['Date'])
                if 'Date' in existing_data.columns:
                    existing_data = existing_data.sort_values('Date')
                    last_date = existing_data['Date'].max()
                    
                    # If data is current (within last 2 days), skip download
                    if pd.to_datetime(last_date).date() >= (datetime.now() - timedelta(days=2)).date():
                        print(f"⏭️  [{i}/{total}] {ticker} is up to date (latest: {last_date.date()})")
                        should_download = False

            if should_download:
                print(f"⬇️  [{i}/{total}] Downloading {ticker}...")
                
                # Download data from yfinance
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                if data.empty:
                    print(f"❌ [{i}/{total}] No data found for {ticker}")
                    continue
                
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                
                # Keep only OHLCV data
                columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
                data = data[[col for col in columns_to_keep if col in data.columns]]
                
                # Sort by date
                data.sort_values('date', inplace=True)
                data.reset_index(drop=True, inplace=True)
                
                # Save or append data
                if file_exists:
                    existing_data = pd.read_csv(filepath, parse_dates=['date'])
                    last_date = existing_data['date'].max()
                    new_data = data[data['date'] > last_date]
                    
                    if not new_data.empty:
                        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                        updated_data.sort_values('date', inplace=True)
                        updated_data.to_csv(filepath, index=False)
                        print(f"✅ [{i}/{total}] {ticker}: Added {len(new_data)} new rows")
                    else:
                        print(f"⏭️  [{i}/{total}] {ticker}: No new data")
                else:
                    data.to_csv(filepath, index=False)
                    print(f"✅ [{i}/{total}] {ticker}: Saved {len(data)} rows")

            # Progress and ETA
            elapsed = time.time() - start_time
            percent = (i / total) * 100
            avg_time = elapsed / i
            eta_seconds = avg_time * (total - i)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            print(f"📊 Progress: {percent:.1f}% | ETA: {eta_str}\n")

        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}\n")

def download_financials(tickers, save_folder="TrainingData/indicators_data/raw/financials"):
    """
    Download financial statements for fundamental analysis
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for ticker in tickers:
        try:
            print(f"📈 Downloading financials for {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Save each statement
            if not income_stmt.empty:
                income_stmt.to_csv(os.path.join(save_folder, f"{ticker}_income.csv"))
            if not balance_sheet.empty:
                balance_sheet.to_csv(os.path.join(save_folder, f"{ticker}_balance.csv"))
            if not cash_flow.empty:
                cash_flow.to_csv(os.path.join(save_folder, f"{ticker}_cashflow.csv"))
            
            # Get key stats
            info = stock.info
            stats_df = pd.DataFrame([info])
            stats_df.to_csv(os.path.join(save_folder, f"{ticker}_info.csv"), index=False)
            
            print(f"✅ Saved financials for {ticker}\n")
            time.sleep(0.5)  # Be nice to the API
            
        except Exception as e:
            print(f"❌ Error downloading financials for {ticker}: {e}\n")

if __name__ == "__main__":
    # Load stock list
    with open('TrainingData/stockList.csv', 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]

    print(f"📋 Loaded {len(tickers)} tickers: {', '.join(tickers)}\n")
    print("=" * 60)
    
    # Download price data
    print("\n🚀 Starting price data download...\n")
    download_yfinance_daily(tickers, period="10y")
    
    # Optionally download financials
    print("\n" + "=" * 60)
    download_choice = input("\n💰 Download financial data too? (y/n): ").lower()
    if download_choice == 'y':
        download_financials(tickers)
    
    print("\n✅ All done!")