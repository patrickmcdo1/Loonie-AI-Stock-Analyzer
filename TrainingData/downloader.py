import subprocess

# Get stock price data (OHLCV)
subprocess.run(["python", "TrainingData/featuresPy/stockScrapper.py"])

# Get market data (SPY/VIX for market context)
subprocess.run(["python", "TrainingData/featuresPy/markets.py"])
