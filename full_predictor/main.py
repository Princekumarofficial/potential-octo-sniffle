from helpers.data_loaders import list_processed_stocks, load_processed_stock

stocks = list_processed_stocks()
data = load_processed_stock("RELIANCE")
subset = load_processed_stock("RELIANCE", columns=["close", "segment_id"])