# CometAgent Confidence Scoring System

## Overview

The CometAgent Confidence Scoring System assigns a 0-100 confidence score to each trading signal based on multiple weighted factors. This helps traders prioritize signals and manage risk more effectively.

## Implementation Summary

### ✅ Completed Features

1. **New `confidence_scorer.py` Module**
   - Standalone confidence calculation engine
   - Weighted scoring system with four components
   - Configurable weights and thresholds
   - Reuses existing indicator functions to avoid code duplication

2. **Command Line Interface Enhancements**
   - `--min-confidence N` flag to filter signals below threshold
   - `--show-confidence-viz` flag to generate distribution plots
   - Confidence scores displayed in console output

3. **Signal Output Format Enhanced**
   - JSON-compatible confidence breakdown
   - Individual component scores included
   - Format: `{"signal": "BUY", "confidence": 78, "breakdown": {...}}`

4. **Backtesting Integration**
   - Confidence scoring in historical backtests
   - Win rate analysis by confidence level
   - Confidence vs. performance correlation metrics

5. **Visualization Tools**
   - Confidence distribution histograms
   - Confidence level zones (High/Medium/Low)
   - Statistical summaries (mean, median, range)

## Confidence Components

### Technical Indicator Agreement (40% weight)
- **RSI**: Oversold/overbought levels and extremes
- **MACD**: Signal line crossovers and momentum
- **Moving Averages**: SMA50/200 trend analysis
- **Bollinger Bands**: Mean reversion signals
- **Agreement Score**: Percentage of indicators supporting the signal direction

### Sentiment Alignment (30% weight)
- **News Sentiment**: Financial news analysis via VADER
- **Social Sentiment**: X/Twitter and Reddit sentiment
- **Alignment Check**: Does sentiment match signal direction?
- **Strength Factor**: How extreme are the sentiment readings?

### Signal Strength (20% weight)
- **RSI Distance**: How far from oversold/overbought thresholds
- **MACD Magnitude**: Size of MACD vs signal line difference
- **Bollinger Position**: Distance from band extremes
- **Normalized Scoring**: Accounts for price and volatility differences

### Historical Reliability (10% weight)
- **Backtest Results**: Win rate for similar signals (if available)
- **Symbol-Specific**: Historical performance for the specific asset
- **Horizon-Aware**: Different reliability for short/medium/long horizons
- **Default Confidence**: 70/100 when no historical data available

## Usage Examples

### Basic Signal Generation with Confidence
```bash
python agentic-trader.py \
  --portfolio-type stocks \
  --horizon short \
  --platform coinbase \
  --file portfolio.csv
```

### Filter by Minimum Confidence
```bash
python agentic-trader.py \
  --portfolio-type crypto \
  --horizon medium \
  --platform coinbase \
  --file crypto-portfolio.csv \
  --min-confidence 70
```

### Generate Confidence Visualization
```bash
python agentic-trader.py \
  --portfolio-type stocks \
  --horizon long \
  --platform fidelity \
  --file stocks.csv \
  --show-confidence-viz
```

### Backtesting with Confidence Analysis
```bash
python backtester.py \
  --symbol BTC-USD \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --horizon short
```

## Signal Output Format

### Console Output
```
=== COMET TRADING REPORT ===
Version: 3.23
Portfolio type: stocks | Horizon: short | Platform: coinbase

=== CONFIDENCE SUMMARY ===
Average Confidence: 68.5
Median Confidence: 71.2
Confidence Range: 45.3 - 89.7
High Confidence (≥70): 5 signals
Medium Confidence (40-69): 3 signals
Low Confidence (<40): 0 signals

symbol    close  total_score  confidence  decision    target_pct_delta
AAPL      150.0        0.65        78.4  Buy                     1.5
TSLA      245.8       -0.42        56.7  Trim                   -1.5
```

### CSV Output
The signals CSV now includes additional confidence columns:
- `confidence`: Overall confidence score (0-100)
- `conf_technical`: Technical indicator component
- `conf_sentiment`: Sentiment alignment component
- `conf_strength`: Signal strength component
- `conf_historical`: Historical reliability component

### JSON API Format
```json
{
  "signal": "BUY",
  "confidence": 78.4,
  "breakdown": {
    "technical": 85.0,
    "sentiment": 70.0,
    "strength": 80.0,
    "historical": 65.0
  }
}
```

## Configuration

### Confidence Weights (in confidence_scorer.py)
```python
self.weights = {
    'technical': 0.40,    # Technical indicator agreement
    'sentiment': 0.30,    # Sentiment alignment
    'strength': 0.20,     # Signal strength
    'historical': 0.10    # Historical reliability
}
```

### RSI Thresholds
```python
self.rsi_oversold = 30
self.rsi_overbought = 70
self.rsi_extreme_oversold = 20
self.rsi_extreme_overbought = 80
```

## Confidence Levels

- **High Confidence (≥70)**: Strong agreement across multiple factors
- **Medium Confidence (40-69)**: Moderate agreement with some conflicting signals
- **Low Confidence (<40)**: Weak or conflicting signals, high uncertainty

## Backtesting Confidence Analysis

The backtesting module now provides:

```
Confidence Analysis:
Average Confidence:       72.3
Median Confidence:        74.1
Confidence Range:         45.2 - 91.8

Confidence vs Performance:
High Confidence (≥70):    78.5% win rate (14 trades)
Medium Confidence (40-69): 62.1% win rate (8 trades)
Low Confidence (<40):     41.7% win rate (3 trades)
```

## Integration Benefits

1. **Risk Management**: Filter out low-confidence signals
2. **Position Sizing**: Scale position sizes based on confidence
3. **Performance Analysis**: Track confidence vs. actual outcomes
4. **Strategy Optimization**: Identify what drives high-confidence signals
5. **Automated Trading**: Use confidence thresholds for automated execution

## Files Modified/Added

### New Files
- `confidence_scorer.py`: Core confidence scoring engine
- `test_confidence_simple.py`: Testing and validation script
- `CONFIDENCE_SCORING_README.md`: This documentation

### Modified Files
- `agentic-trader.py`: Integrated confidence scoring and CLI flags
- `backtester.py`: Added confidence analysis to backtesting reports

## Testing

Run the test suite to verify functionality:
```bash
python test_confidence_simple.py
```

Expected output shows confidence calculations for different signal scenarios with breakdown by component.

## Future Enhancements

1. **Machine Learning**: Train models on historical confidence vs. outcomes
2. **Real-time Adaptation**: Adjust weights based on recent performance
3. **Sector-Specific**: Different confidence models for different asset classes
4. **Alternative Data**: Integrate additional data sources for confidence calculation
5. **API Integration**: RESTful API for confidence scoring as a service

## Disclaimer

The confidence scoring system is a risk management tool and not a guarantee of trading success. All trading involves risk, and past performance does not guarantee future results. Users should conduct their own analysis and risk assessment before making trading decisions.