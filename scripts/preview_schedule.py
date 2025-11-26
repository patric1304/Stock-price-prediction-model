"""
Preview what stocks will be trained on different days
"""
from datetime import datetime, timedelta

STOCK_GROUPS = {
    # Week 1
    "thursday": ["ROP", "RSG", "RTX", "RVTY"],
    "friday": ["SBAC", "SBUX", "SCHW", "SEE"],
    "saturday": ["SHW", "SJM", "SLB", "SNA"],
    "sunday": ["SO", "SPG", "SPGI", "SRE"],
    "monday": ["STE", "STZ", "SWK", "SWKS"],
    "tuesday": ["SYF", "SYK", "SYY", "T"],
    "wednesday": ["TAP", "TDY", "TECH", "TEL"],
    # Week 2
    "thursday2": ["TER", "TFC", "TFX", "TGT"],
    "friday2": ["TJX", "TMUS", "TPR", "TRGP"],
    "saturday2": ["TRMB", "TROW", "TRV", "TSLA"],
    "sunday2": ["TSN", "TT", "TTWO", "TXN"],
    "monday2": ["TXT", "TYL", "UDR", "UL"],
    "tuesday2": ["UMPQ", "UNH", "UNP", "UPS"],
    "wednesday2": ["URI", "USB", "VEV", "VFC"],
    # Week 3
    "thursday3": ["VLO", "VMC", "VRSK", "VRSN"],
    "friday3": ["VRTX", "VTR", "VZ", "WBA"],
    "saturday3": ["WBD", "WEC", "WELL", "WFC"],
    "sunday3": ["WHR", "WM", "WMB", "WMT"],
    "monday3": ["WRB", "WRK", "WST", "WY"],
    "tuesday3": ["WYNN", "XEL", "XOM", "XRAY"],
    "wednesday3": ["ZBH", "ZBRA", "ZION", "ZTS"]
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
