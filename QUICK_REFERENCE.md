# Quick Reference Guide
## Enhanced Nifty Option Prediction System v1.0.0

**Trading Quick Reference Card**

---

## 🚀 Quick Start Checklist

### Daily Routine
- [ ] **9:00 AM**: Check system status, load fresh data
- [ ] **9:15 AM**: Train models with overnight data
- [ ] **9:30 AM**: Generate prediction, review confidence
- [ ] **9:45 AM**: Select strategy based on prediction
- [ ] **10:00 AM**: Execute trades with proper position sizing
- [ ] **3:30 PM**: Review positions, adjust if needed
- [ ] **3:45 PM**: Close positions or roll if necessary

---

## 📊 Prediction Interpretation

| Prediction | Confidence | Action | Strategy |
|------------|------------|--------|----------|
| 🟢 BULLISH | High (70%+) | Strong Buy | Long Iron Condor |
| 🟢 BULLISH | Medium (50-70%) | Moderate Buy | Bull Call Spread |
| 🟢 BULLISH | Low (30-50%) | Weak Buy | Small Bull Spread |
| 🔴 BEARISH | High (70%+) | Strong Sell | Short Iron Condor |
| 🔴 BEARISH | Medium (50-70%) | Moderate Sell | Bear Put Spread |
| 🔴 BEARISH | Low (30-50%) | Weak Sell | Small Bear Spread |
| 🟡 SIDEWAYS | High (70%+) | Range Trade | Short Iron Condor |
| 🟡 SIDEWAYS | Medium (50-70%) | Neutral | Iron Condor |
| 🟡 SIDEWAYS | Low (30-50%) | Avoid Trade | Wait for clarity |

---

## 📈 VIX Trading Guide

| VIX Level | Market Condition | Strategy Preference | Position Size |
|-----------|------------------|-------------------|---------------|
| < 15 | Low Volatility | Short strategies | Normal |
| 15-25 | Normal Volatility | Balanced strategies | Normal |
| 25-35 | High Volatility | Long strategies | Reduced |
| > 35 | Extreme Volatility | Avoid new positions | Minimal |

---

## ⏰ Best Trading Times

### Intraday (Same Day Expiry)
- **Best**: Tuesday-Thursday, 10:30-11:30 AM, 2:30-3:30 PM
- **Avoid**: Monday (gaps), Friday (weekend risk), 9:15-10:30 AM (high volatility)

### Weekly Expiry
- **Best**: Monday-Wednesday
- **Avoid**: Thursday-Friday (time decay acceleration)

### Monthly Expiry
- **Best**: First 2 weeks of month
- **Avoid**: Last week (accelerated decay)

---

## 💰 Position Sizing Rules

### Conservative (1-2% risk)
- Capital: ₹10,00,000
- Risk per trade: ₹10,000-20,000
- Max positions: 5-10

### Moderate (2-5% risk)
- Capital: ₹10,00,000
- Risk per trade: ₹20,000-50,000
- Max positions: 3-5

### Aggressive (5-10% risk)
- Capital: ₹10,00,000
- Risk per trade: ₹50,000-1,00,000
- Max positions: 1-3

---

## 🎯 Option Strategies Quick Guide

### Bullish Strategies
| Strategy | When to Use | Max Profit | Max Loss | Breakeven |
|----------|-------------|------------|----------|-----------|
| **Long Iron Condor** | High confidence bullish | Premium received | Limited | Two points |
| **Bull Call Spread** | Moderate bullish | Spread - Premium | Premium paid | Strike + Premium |
| **Long Call** | Very bullish | Unlimited | Premium paid | Strike + Premium |

### Bearish Strategies
| Strategy | When to Use | Max Profit | Max Loss | Breakeven |
|----------|-------------|------------|----------|-----------|
| **Short Iron Condor** | High confidence bearish | Premium received | Limited | Two points |
| **Bear Put Spread** | Moderate bearish | Spread - Premium | Premium paid | Strike - Premium |
| **Long Put** | Very bearish | Strike - Premium | Premium paid | Strike - Premium |

### Sideways Strategies
| Strategy | When to Use | Max Profit | Max Loss | Breakeven |
|----------|-------------|------------|----------|-----------|
| **Short Iron Condor** | Range-bound market | Premium received | Limited | Two points |
| **Short Straddle** | Low volatility expected | Premium received | Unlimited | Strike ± Premium |
| **Short Strangle** | Range-bound, low vol | Premium received | Unlimited | Two points |

---

## ⚠️ Risk Management Rules

### Stop Loss Rules
- **Intraday**: 50% of premium
- **Weekly**: 30% of premium
- **Monthly**: 25% of premium

### Profit Taking Rules
- **Intraday**: 50-75% of premium
- **Weekly**: 60-80% of premium
- **Monthly**: 70-90% of premium

### Position Limits
- Never risk more than 10% of capital
- Maximum 5 positions at once
- Avoid highly correlated positions
- Reduce size during high VIX periods

---

## 🔧 System Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Insufficient data" | Increase historical period to 90+ days |
| "HAR-RV not training" | Check data quality, retry training |
| "Low confidence" | Refresh data, retrain models |
| "Poor backtest" | Review strategy selection, market conditions |

### Data Refresh
- **Automatic**: Every hour
- **Manual**: Click "Force Refresh Data"
- **Cache**: Cleared automatically on refresh

---

## 📱 UI Navigation

### Main Sections
1. **Market Data & Analysis**: Current prices, VIX, indicators
2. **Model Training & Performance**: Model status, training results
3. **Prediction & Strategy**: Forecast, confidence, recommended strategy
4. **Data Visualization**: Charts, technical analysis
5. **Position Sizing**: Risk calculation, position recommendations
6. **Strategy Backtest**: Historical performance validation

### Key Buttons
- **Load Market Data**: Fetch fresh data
- **Train Models**: Train AI models
- **Generate Prediction**: Get market forecast
- **Run Backtest**: Validate strategy

---

## 📊 Performance Metrics

### Good Performance Indicators
- **Sharpe Ratio**: > 1.0
- **Max Drawdown**: < 10%
- **Win Rate**: > 55%
- **Monthly Returns**: Consistent positive

### Red Flags
- **Sharpe Ratio**: < 0.5
- **Max Drawdown**: > 20%
- **Win Rate**: < 40%
- **Monthly Returns**: Highly volatile

---

## 🎓 Learning Resources

### Understanding Models
- **TFT**: Multi-horizon forecasting with attention
- **HAR-RV**: Advanced volatility modeling
- **ML Ensemble**: Combines multiple signals

### Option Basics
- Learn Greeks (Delta, Gamma, Theta, Vega)
- Understand time decay
- Study volatility impact
- Practice with paper trading

### Risk Management
- Position sizing principles
- Stop loss strategies
- Portfolio diversification
- Emotional discipline

---

## 📞 Support & Help

### Documentation
- **User Manual**: Complete system guide
- **Technical Docs**: Developer reference
- **Quick Reference**: This card

### Best Practices
1. **Start Small**: Begin with small positions
2. **Paper Trade**: Practice before real money
3. **Keep Records**: Track all trades and results
4. **Continuous Learning**: Stay updated with market changes
5. **Risk First**: Always prioritize risk management

---

**Remember**: This system is a tool to assist your trading decisions. Always do your own analysis and never risk more than you can afford to lose.

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Author**: Pravin A Mathew
