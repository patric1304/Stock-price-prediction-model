"""
Preview what stocks will be trained on different days
"""
from datetime import datetime, timedelta

STOCK_GROUPS = {
    "thursday": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "friday": ["NVDA", "META", "TSLA", "BRK.B"],
    "saturday": ["UNH", "XOM", "JNJ", "JPM"],
    "sunday": ["V", "PG", "MA", "HD"],
    "monday": ["CVX", "MRK", "ABBV", "KO"],
    "tuesday": ["PEP", "AVGO", "COST", "WMT"],
    "wednesday": ["MCD", "CSCO", "ACN", "TMO"],
    "thursday2": ["NFLX", "ABT", "CRM", "ORCL"],
    "friday2": ["NKE", "INTC", "VZ", "CMCSA"],
    "saturday2": ["AMD", "QCOM", "PM", "LLY"],
    "sunday2": ["ADBE", "DHR", "TXN", "NEE"],
    "monday2": ["UNP", "RTX", "INTU", "HON"],
    "tuesday2": ["CAT", "LOW", "BA", "GS"],
    "wednesday2": ["SPGI", "BLK", "AXP", "SBUX"]
}

day_names = ["thursday", "friday", "saturday", "sunday", "monday", "tuesday", "wednesday",
             "thursday2", "friday2", "saturday2", "sunday2", "monday2", "tuesday2", "wednesday2"]

print("🗓️  14-DAY TRAINING SCHEDULE")
print("=" * 70)

start_date = datetime(2025, 11, 7)
total_stocks = 0

for i in range(14):
    current_date = start_date + timedelta(days=i)
    day_key = day_names[i]
    stocks = STOCK_GROUPS[day_key]
    total_stocks += len(stocks)
    
    print(f"\nDay {i+1:2d} ({current_date.strftime('%Y-%m-%d')} - {day_key.capitalize()})")
    print(f"  Stocks: {', '.join(stocks)}")
    print(f"  Running total: {total_stocks} stocks")
    
    if i == 0:
        print(f"  Samples: ~148 (4 stocks × 37 days)")
    else:
        print(f"  Samples: ~{total_stocks * 37} ({total_stocks} stocks × 37 days)")

print("\n" + "=" * 70)
print(f"TOTAL: {total_stocks} stocks after 14 days")
print(f"Expected final samples: ~{total_stocks * 37}")
