#!/usr/bin/env python3
"""
Simple test to verify confidence scoring integration
"""

import sys
sys.path.append('.')

from confidence_scorer import ConfidenceScorer, ConfidenceBreakdown

def test_confidence_scoring():
    print("Testing CometAgent Confidence Scoring System")
    print("=" * 50)

    # Create test data
    test_indicators = {
        'close': 150.0,
        'rsi14': 25.0,  # Oversold (bullish signal)
        'macd': 0.5,    # Above signal line (bullish)
        'macd_signal': 0.3,
        'sma50': 145.0,   # Close above SMA50 (bullish)
        'sma200': 140.0,  # Close above SMA200 (bullish)
        'bb_up': 155.0,
        'bb_dn': 145.0,
        'bb_mid': 150.0,
        'atr14': 2.5
    }

    test_sentiment = {
        'sentiment_score': 0.3,    # Positive sentiment
        'x_only_score': 0.4,       # Positive social sentiment
        'reddit_only_score': 0.2   # Positive reddit sentiment
    }

    # Test confidence calculation
    scorer = ConfidenceScorer()

    print("Test Case 1: Strong Buy Signal")
    print("-" * 30)
    confidence = scorer.calculate_confidence(
        test_indicators,
        test_sentiment,
        'Buy',
        'AAPL',
        'short'
    )

    print(scorer.format_confidence_output(confidence, 'AAPL', 'Buy'))

    # Test case 2: Sell signal with negative sentiment
    test_sentiment_negative = {
        'sentiment_score': -0.4,
        'x_only_score': -0.3,
        'reddit_only_score': -0.5
    }

    test_indicators_bearish = test_indicators.copy()
    test_indicators_bearish['rsi14'] = 85.0  # Overbought
    test_indicators_bearish['close'] = 135.0  # Below moving averages

    print("\nTest Case 2: Sell Signal")
    print("-" * 30)
    confidence_sell = scorer.calculate_confidence(
        test_indicators_bearish,
        test_sentiment_negative,
        'Sell',
        'AAPL',
        'short'
    )

    print(scorer.format_confidence_output(confidence_sell, 'AAPL', 'Sell'))

    # Test JSON output format
    print("\nJSON Format Output:")
    print("-" * 20)
    signal_with_confidence = {
        "signal": "BUY",
        "confidence": confidence.overall,
        "breakdown": {
            "technical": confidence.technical,
            "sentiment": confidence.sentiment,
            "strength": confidence.strength,
            "historical": confidence.historical
        }
    }

    import json
    print(json.dumps(signal_with_confidence, indent=2))

    print("\n" + "=" * 50)
    print("Confidence scoring test completed successfully!")

if __name__ == "__main__":
    test_confidence_scoring()