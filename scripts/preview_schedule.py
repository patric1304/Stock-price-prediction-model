"""
Preview what stocks will be trained on different days
"""
from datetime import datetime, timedelta

STOCK_GROUPS = {
    # Week 1
    "thursday": ["NVDA", "AMD", "INTC", "MU"],
    "friday": ["JPM", "BAC", "WFC", "C"],
    "saturday": ["JNJ", "PFE", "MRK", "ABBV"],
    "sunday": ["KO", "PEP", "MCD", "SBUX"],
    "monday": ["XOM", "CVX", "COP", "SLB"],
    "tuesday": ["WMT", "TGT", "COST", "HD"],
    "wednesday": ["UNH", "ELV", "CVS", "CI"],
    # Week 2
    "thursday2": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "friday2": ["TSLA", "F", "GM", "TM"],
    "saturday2": ["DIS", "NFLX", "CMCSA", "WBD"],
    "sunday2": ["V", "MA", "AXP", "PYPL"],
    "monday2": ["BA", "LMT", "RTX", "GD"],
    "tuesday2": ["CAT", "DE", "HON", "GE"],
    "wednesday2": ["NEE", "DUK", "SO", "D"],
    # Week 3
    "thursday3": ["META", "SNAP", "PINS", "TTD"],
    "friday3": ["CRM", "ADBE", "ORCL", "SAP"],
    "saturday3": ["NKE", "LULU", "ADDYY", "UAA"],
    "sunday3": ["BKNG", "EXPE", "ABNB", "MAR"],
    "monday3": ["UPS", "FDX", "DAL", "UAL"],
    "tuesday3": ["GS", "MS", "BLK", "SCHW"],
    "wednesday3": ["IBM", "CSCO", "HPE", "DELL"]
}

day_names = [
    "thursday", "friday", "saturday", "sunday", "monday", "tuesday", "wednesday",
    "thursday2", "friday2", "saturday2", "sunday2", "monday2", "tuesday2", "wednesday2",
    "thursday3", "friday3", "saturday3", "sunday3", "monday3", "tuesday3", "wednesday3"
]

print("🗓️  21-DAY TRAINING SCHEDULE")
print("=" * 70)

start_date = datetime(2025, 11, 21)
total_stocks = 0

for i in range(21):
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
print(f"TOTAL: {total_stocks} stocks after 21 days")
print(f"Expected final samples: ~{total_stocks * 37}")
