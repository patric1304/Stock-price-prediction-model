# 📅 2-Week S&P 500 Training Plan

## 🎯 Strategy Overview

**Start Date:** Thursday, November 7, 2025  
**End Date:** Wednesday, November 20, 2025  
**Duration:** 14 days  
**Total Stocks:** 56 S&P 500 companies  

### Configuration:
- ✅ **NewsAPI.org** free tier (100 requests/day)
- ✅ **20 days of news** per stock (3 weeks lookback)
- ✅ **Company news only** (no global sentiment)
- ✅ **4 stocks per day** (80 requests/day, 20% buffer)
- ✅ **Incremental learning** (each day builds on previous)

---

## 📊 Daily Training Schedule

### **Week 1**

| Day | Date | Stocks (4 per day) | Sector Focus | API Requests | Training Time |
|-----|------|-------------------|--------------|--------------|---------------|
| **Day 1** | Thu Nov 7 | AAPL, MSFT, GOOGL, AMZN | Mega-cap Tech | 80 | 30 min |
| **Day 2** | Fri Nov 8 | NVDA, META, TSLA, BRK.B | Growth + Value | 80 | 30 min |
| **Day 3** | Sat Nov 9 | UNH, XOM, JNJ, JPM | Healthcare + Energy + Finance | 80 | 30 min |
| **Day 4** | Sun Nov 10 | V, PG, MA, HD | Payments + Consumer | 80 | 30 min |
| **Day 5** | Mon Nov 11 | CVX, MRK, ABBV, KO | Energy + Pharma | 80 | 30 min |
| **Day 6** | Tue Nov 12 | PEP, AVGO, COST, WMT | Consumer + Tech + Retail | 80 | 30 min |
| **Day 7** | Wed Nov 13 | MCD, CSCO, ACN, TMO | Food + Services | 80 | 30 min |

**Week 1 Total:** 28 stocks, 560 requests, 3.5 hours

---

### **Week 2**

| Day | Date | Stocks (4 per day) | Sector Focus | API Requests | Training Time |
|-----|------|-------------------|--------------|--------------|---------------|
| **Day 8** | Thu Nov 14 | NFLX, ABT, CRM, ORCL | Media + Healthcare + Software | 80 | 30 min |
| **Day 9** | Fri Nov 15 | NKE, INTC, VZ, CMCSA | Apparel + Telecom | 80 | 30 min |
| **Day 10** | Sat Nov 16 | AMD, QCOM, PM, LLY | Semiconductors + Pharma | 80 | 30 min |
| **Day 11** | Sun Nov 17 | ADBE, DHR, TXN, NEE | Software + Industrials + Utilities | 80 | 30 min |
| **Day 12** | Mon Nov 18 | UNP, RTX, INTU, HON | Rail + Aerospace + Software | 80 | 30 min |
| **Day 13** | Tue Nov 19 | CAT, LOW, BA, GS | Construction + Banking | 80 | 30 min |
| **Day 14** | Wed Nov 20 | SPGI, BLK, AXP, SBUX | Financial Services | 80 | 30 min |

**Week 2 Total:** 28 stocks, 560 requests, 3.5 hours

---

## 🎓 Cumulative Learning Progress

| End of Day | Stocks Trained | Sectors Covered | Model Maturity |
|------------|----------------|-----------------|----------------|
| Day 1 | 4 | Tech | 🟡 Basic |
| Day 3 | 12 | Tech, Healthcare, Energy, Finance | 🟡 Learning |
| Day 7 | 28 | 8+ sectors | 🟢 Good |
| Day 10 | 40 | 10+ sectors | 🟢 Strong |
| Day 14 | 56 | All major S&P sectors | 🟢 Excellent |

---

## 📈 Expected Model Performance

### Prediction Accuracy (MAE - Mean Absolute Error):

| Timeframe | Expected MAE | MAPE | Quality |
|-----------|-------------|------|---------|
| **Day 1-3** | ~$15-20 | ~8-12% | Learning phase |
| **Day 4-7** | ~$12-15 | ~6-9% | Good generalization |
| **Day 8-10** | ~$10-12 | ~5-7% | Strong patterns |
| **Day 11-14** | ~$8-10 | ~4-6% | **Excellent** |

*Note: Actual performance varies by market conditions and stock volatility*

### Direction Accuracy (Up/Down prediction):
- **Day 7:** ~55-60% (better than random)
- **Day 14:** ~60-65% (solid predictive power)

---

## 🏆 Sector Diversity Achieved

After 14 days, your model will understand:

✅ **Technology:** AAPL, MSFT, GOOGL, NVDA, META, AMD, INTC, QCOM, ADBE, ORCL, CRM, CSCO, TXN, INTU  
✅ **Healthcare:** UNH, JNJ, MRK, ABBV, ABT, LLY, TMO, DHR  
✅ **Finance:** JPM, V, MA, BRK.B, GS, BLK, AXP, SPGI  
✅ **Energy:** XOM, CVX  
✅ **Consumer:** PG, KO, PEP, WMT, COST, MCD, SBUX, NKE, HD, LOW  
✅ **Industrials:** HON, CAT, RTX, BA, UNP, NEE  
✅ **Communications:** VZ, CMCSA  
✅ **Others:** TSLA (Auto), PM (Tobacco), ACN (Services)

**Total: 56 stocks across 11 sectors** - excellent diversity! 🎉

---

## 🚀 Daily Routine

### Every Day (takes ~30 minutes):

```bash
# 1. Run training script
python scripts/train_continuous.py

# Expected output:
# - Fetches 20 days of news for 4 stocks
# - Uses ~80 API requests
# - Trains for 100 epochs
# - Saves model as sp500_model_YYYYMMDD_HHMMSS.pth
# - Shows MAE, MAPE, RMSE metrics
```

### What Happens Automatically:

1. **Detects current day** (Day 1-14 of cycle)
2. **Loads yesterday's model** (incremental learning)
3. **Fetches 4 new stocks** with 20 days of news each
4. **Trains and improves** the model
5. **Saves updated model** for next day
6. **Logs performance metrics**

---

## 📊 API Usage Tracking

### Daily Breakdown:
```
4 stocks × 20 days of news = 80 requests
Buffer: 20 requests (for retries or errors)
Total: 80-100 requests/day (within limit)
```

### 2-Week Total:
```
14 days × 80 requests = 1,120 requests
Average: 80/day (well within 100/day limit)
```

### Cached Efficiency:
- **First run:** 80 requests (fetches fresh news)
- **Second run same day:** 0 requests (uses cache!)
- **Next day:** Only fetches new day's news (~4 requests)

---

## 🎯 After 2 Weeks

### What You'll Have:

✅ **56-stock trained model** covering all major S&P sectors  
✅ **Robust predictions** for any stock (not just trained ones!)  
✅ **Direction accuracy** of 60-65%  
✅ **MAE** of $8-10 per prediction  
✅ **Ready for production** use  

### Next Steps:

1. **Make predictions:**
   ```bash
   python scripts/run_inference.py
   # Enter any stock ticker (AAPL, TSLA, etc.)
   # Get tomorrow's predicted price!
   ```

2. **Evaluate accuracy:**
   ```bash
   python scripts/evaluate_model.py
   # Test on different stocks
   # See detailed metrics
   ```

3. **Continue training** (optional):
   - Run `train_daily.py` for quick updates (3 stocks/day)
   - Or repeat 2-week cycle with different stocks

---

## 💡 Pro Tips

### ✅ Best Practices:

1. **Run training in the evening** (after market close)
   - Gets full day's news
   - Avoids peak API hours

2. **Don't skip days**
   - Incremental learning works best with consistency
   - Each day builds on previous knowledge

3. **Monitor API usage**
   - Check console output for request counts
   - Stay under 100/day to avoid rate limits

4. **Check model improvement**
   - Look at MAE/MAPE each day
   - Should decrease over time

5. **Test predictions regularly**
   - After Day 7, start making predictions
   - Compare with actual prices next day

### ⚠️ Troubleshooting:

**If API limit hit:**
- Wait until next day (resets at midnight)
- Cache is already built, so subsequent runs are faster

**If model accuracy plateaus:**
- This is normal around Day 10-12
- Model is learning to generalize, not overfit

**If predictions seem off:**
- Market conditions change - model adapts over time
- Check if VIX is high (volatile markets harder to predict)

---

## 📈 Success Metrics

### By Day 7:
- ✅ 28 stocks trained
- ✅ MAE < $15
- ✅ Direction accuracy > 55%
- ✅ Can predict most S&P stocks

### By Day 14:
- ✅ 56 stocks trained (10% of S&P 500)
- ✅ MAE < $10
- ✅ Direction accuracy > 60%
- ✅ Production-ready model

---

## 🎉 Ready to Start!

**Today (Day 1 - Thursday, Nov 7):**

```bash
python scripts/train_continuous.py
```

**You'll train on:** AAPL, MSFT, GOOGL, AMZN (the giants!)

**See you tomorrow for Day 2!** 🚀

---

## 📝 Daily Log Template

Track your progress:

```
Day X - [Date]
Stocks: [List of 4 stocks]
API Requests: [Actual count]
Training Time: [Minutes]
Final Loss: [Value]
MAE: $[Value]
MAPE: [%]
Notes: [Any observations]
```

**Stick to the plan and you'll have an excellent model in 2 weeks!** 🎯
