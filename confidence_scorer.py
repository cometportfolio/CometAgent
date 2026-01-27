#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
confidence_scorer.py
Confidence scoring system for CometAgent trading signals.

Assigns a 0-100 confidence score to each trading signal based on:
- Technical Indicator Agreement (40%): How many indicators align with signal direction
- Sentiment Alignment (30%): Agreement between news/social sentiment and strength
- Signal Strength (20%): Distance from threshold values (e.g., RSI extreme levels)
- Historical Reliability (10%): Historical win rate for similar setups (if backtesting available)

Usage:
from confidence_scorer import ConfidenceScorer
scorer = ConfidenceScorer()
confidence = scorer.calculate_confidence(indicators, sentiment_scores, signal_direction, symbol)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os


@dataclass
class ConfidenceBreakdown:
    """Breakdown of confidence score components"""
    technical: float     # 0-100: Technical indicator agreement
    sentiment: float     # 0-100: Sentiment alignment and strength
    strength: float      # 0-100: Signal strength (distance from thresholds)
    historical: float    # 0-100: Historical reliability
    overall: float       # 0-100: Weighted final confidence score


class ConfidenceScorer:
    """
    Calculates confidence scores for trading signals using multiple factors.
    Reuses existing functions from agentic-trader.py to avoid duplication.
    """

    def __init__(self):
        # Weights for confidence components (must sum to 1.0)
        self.weights = {
            'technical': 0.40,    # Technical indicator agreement
            'sentiment': 0.30,    # Sentiment alignment
            'strength': 0.20,     # Signal strength
            'historical': 0.10    # Historical reliability
        }

        # RSI thresholds for signal strength calculation
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_extreme_oversold = 20
        self.rsi_extreme_overbought = 80

        # Historical win rate cache (symbol -> win_rate)
        self._historical_cache = {}

    def calculate_confidence(self,
                           indicators: Dict,  # From agentic-trader._indicators()
                           sentiment_scores: Dict,  # From agentic-trader sentiment functions
                           signal_direction: str,  # 'Buy', 'Sell', 'Hold', etc.
                           symbol: str,
                           horizon: str = 'short') -> ConfidenceBreakdown:
        """
        Calculate overall confidence score and component breakdown.

        Args:
            indicators: Dictionary with RSI, MACD, Bollinger, SMA values
            sentiment_scores: Dictionary with sentiment analysis results
            signal_direction: Trading signal ('Buy', 'Sell', 'Hold', etc.)
            symbol: Trading symbol for historical analysis
            horizon: Trading horizon ('short', 'medium', 'long')

        Returns:
            ConfidenceBreakdown with component scores and overall confidence
        """
        # Calculate individual components
        technical_score = self._calculate_technical_confidence(indicators, signal_direction, horizon)
        sentiment_score = self._calculate_sentiment_confidence(sentiment_scores, signal_direction)
        strength_score = self._calculate_strength_confidence(indicators, signal_direction)
        historical_score = self._calculate_historical_confidence(symbol, signal_direction, horizon)

        # Calculate weighted overall score
        overall_score = (
            self.weights['technical'] * technical_score +
            self.weights['sentiment'] * sentiment_score +
            self.weights['strength'] * strength_score +
            self.weights['historical'] * historical_score
        )

        return ConfidenceBreakdown(
            technical=round(technical_score, 1),
            sentiment=round(sentiment_score, 1),
            strength=round(strength_score, 1),
            historical=round(historical_score, 1),
            overall=round(overall_score, 1)
        )

    def _calculate_technical_confidence(self, indicators: Dict, signal_direction: str, horizon: str) -> float:
        """
        Calculate confidence based on technical indicator agreement (40% weight).

        Checks alignment of RSI, MACD, Bollinger Bands, and moving averages
        with the signal direction. Higher alignment = higher confidence.
        """
        if signal_direction in ['Hold', 'No Data']:
            return 50.0  # Neutral confidence for hold signals

        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # RSI Analysis
        rsi = indicators.get('rsi14', 50)
        if not np.isnan(rsi):
            total_signals += 1
            if rsi < self.rsi_oversold:  # Oversold = bullish
                bullish_signals += 1
            elif rsi > self.rsi_overbought:  # Overbought = bearish
                bearish_signals += 1

        # MACD Analysis
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if not (np.isnan(macd) or np.isnan(macd_signal)):
            total_signals += 1
            if macd > macd_signal:  # MACD above signal = bullish
                bullish_signals += 1
            else:  # MACD below signal = bearish
                bearish_signals += 1

        # Moving Average Trend Analysis
        close = indicators.get('close', 0)
        sma50 = indicators.get('sma50', 0)
        sma200 = indicators.get('sma200', np.nan)

        if not (np.isnan(close) or np.isnan(sma50)):
            total_signals += 1
            if close > sma50:  # Above SMA50 = bullish
                bullish_signals += 1
            else:  # Below SMA50 = bearish
                bearish_signals += 1

        if not (np.isnan(close) or np.isnan(sma200)):
            total_signals += 1
            if close > sma200:  # Above SMA200 = bullish
                bullish_signals += 1
            else:  # Below SMA200 = bearish
                bearish_signals += 1

        # Bollinger Band Analysis
        bb_up = indicators.get('bb_up', 0)
        bb_dn = indicators.get('bb_dn', 0)
        if not (np.isnan(close) or np.isnan(bb_up) or np.isnan(bb_dn)):
            total_signals += 1
            if close < bb_dn:  # Below lower band = oversold/bullish
                bullish_signals += 1
            elif close > bb_up:  # Above upper band = overbought/bearish
                bearish_signals += 1

        if total_signals == 0:
            return 50.0  # No data available

        # Calculate agreement based on signal direction
        if signal_direction in ['Buy', 'Strong Buy']:
            agreement_ratio = bullish_signals / total_signals
        elif signal_direction in ['Sell', 'Trim']:
            agreement_ratio = bearish_signals / total_signals
        else:
            return 50.0  # Unknown signal type

        # Convert agreement ratio to 0-100 scale
        # 100% agreement = 100 confidence, 0% agreement = 0 confidence
        confidence = agreement_ratio * 100

        # Boost confidence for strong agreement in longer horizons
        if horizon == 'long' and agreement_ratio >= 0.8:
            confidence = min(100, confidence * 1.1)

        return confidence

    def _calculate_sentiment_confidence(self, sentiment_scores: Dict, signal_direction: str) -> float:
        """
        Calculate confidence based on sentiment alignment and strength (30% weight).

        Considers agreement between news sentiment, social sentiment,
        and alignment with signal direction.
        """
        if signal_direction in ['Hold', 'No Data']:
            return 50.0

        # Extract sentiment scores (from agentic-trader sentiment analysis)
        news_sentiment = sentiment_scores.get('sentiment_score', 0)  # -1 to +1
        social_sentiment = sentiment_scores.get('x_only_score', 0)   # -1 to +1
        reddit_sentiment = sentiment_scores.get('reddit_only_score', 0)  # -1 to +1

        # Calculate sentiment agreement (how well do different sources align?)
        sentiments = [s for s in [news_sentiment, social_sentiment, reddit_sentiment] if not np.isnan(s)]
        if len(sentiments) < 2:
            sentiment_agreement = 50.0  # Not enough data
        else:
            # Calculate standard deviation of sentiments (lower = more agreement)
            sentiment_std = np.std(sentiments)
            # Convert std to agreement score (0 std = 100% agreement, high std = low agreement)
            sentiment_agreement = max(0, 100 - (sentiment_std * 100))

        # Calculate sentiment strength (how extreme are the sentiment values?)
        avg_sentiment = np.mean([abs(s) for s in sentiments]) if sentiments else 0
        sentiment_strength = min(100, avg_sentiment * 100)  # Convert to 0-100 scale

        # Calculate signal alignment (does sentiment match signal direction?)
        blended_sentiment = np.mean(sentiments) if sentiments else 0
        if signal_direction in ['Buy', 'Strong Buy']:
            signal_alignment = max(0, blended_sentiment * 100)  # Positive sentiment = good
        elif signal_direction in ['Sell', 'Trim']:
            signal_alignment = max(0, -blended_sentiment * 100)  # Negative sentiment = good
        else:
            signal_alignment = 50.0

        # Combine agreement, strength, and alignment
        # 40% agreement + 30% strength + 30% alignment
        confidence = (0.4 * sentiment_agreement +
                     0.3 * sentiment_strength +
                     0.3 * signal_alignment)

        return min(100, max(0, confidence))

    def _calculate_strength_confidence(self, indicators: Dict, signal_direction: str) -> float:
        """
        Calculate confidence based on signal strength (20% weight).

        Measures how far indicators are from their threshold values.
        Stronger signals (RSI at 25 vs 29 for oversold) get higher confidence.
        """
        if signal_direction in ['Hold', 'No Data']:
            return 50.0

        strength_scores = []

        # RSI strength analysis
        rsi = indicators.get('rsi14', 50)
        if not np.isnan(rsi):
            if signal_direction in ['Buy', 'Strong Buy']:
                # For buy signals, lower RSI = stronger signal
                if rsi <= self.rsi_extreme_oversold:
                    rsi_strength = 100  # Extremely oversold
                elif rsi <= self.rsi_oversold:
                    # Linear scale from oversold threshold to extreme
                    rsi_strength = 60 + 40 * (self.rsi_oversold - rsi) / (self.rsi_oversold - self.rsi_extreme_oversold)
                else:
                    # Above oversold threshold, lower confidence
                    rsi_strength = max(0, 60 * (50 - rsi) / (50 - self.rsi_oversold))
            else:  # Sell signals
                # For sell signals, higher RSI = stronger signal
                if rsi >= self.rsi_extreme_overbought:
                    rsi_strength = 100  # Extremely overbought
                elif rsi >= self.rsi_overbought:
                    # Linear scale from overbought threshold to extreme
                    rsi_strength = 60 + 40 * (rsi - self.rsi_overbought) / (self.rsi_extreme_overbought - self.rsi_overbought)
                else:
                    # Below overbought threshold, lower confidence
                    rsi_strength = max(0, 60 * (rsi - 50) / (self.rsi_overbought - 50))

            strength_scores.append(rsi_strength)

        # MACD strength analysis
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if not (np.isnan(macd) or np.isnan(macd_signal)):
            macd_diff = abs(macd - macd_signal)
            # Normalize MACD difference (larger difference = stronger signal)
            # Use close price to normalize (typical MACD values are small)
            close = indicators.get('close', 100)
            if close > 0:
                normalized_diff = (macd_diff / close) * 10000  # Scale up for visibility
                macd_strength = min(100, normalized_diff * 20)  # Convert to 0-100
                strength_scores.append(macd_strength)

        # Bollinger Band position strength
        close = indicators.get('close', 0)
        bb_up = indicators.get('bb_up', 0)
        bb_dn = indicators.get('bb_dn', 0)
        bb_mid = indicators.get('bb_mid', 0)

        if not any(np.isnan(x) for x in [close, bb_up, bb_dn, bb_mid]):
            bb_width = bb_up - bb_dn
            if bb_width > 0:
                if close < bb_dn:  # Below lower band
                    bb_strength = min(100, (bb_dn - close) / bb_width * 200)
                elif close > bb_up:  # Above upper band
                    bb_strength = min(100, (close - bb_up) / bb_width * 200)
                else:  # Within bands
                    # Distance from middle band
                    bb_strength = min(100, abs(close - bb_mid) / (bb_width / 2) * 50)

                strength_scores.append(bb_strength)

        # Average all strength scores
        if strength_scores:
            return np.mean(strength_scores)
        else:
            return 50.0  # No strength data available

    def _calculate_historical_confidence(self, symbol: str, signal_direction: str, horizon: str) -> float:
        """
        Calculate confidence based on historical reliability (10% weight).

        If backtesting data is available, uses historical win rate for similar setups.
        Otherwise returns neutral confidence.
        """
        # Check cache first
        cache_key = f"{symbol}_{signal_direction}_{horizon}"
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        # Try to load backtesting results if available
        try:
            # Look for recent backtest files
            backtest_pattern = f"backtest_results_{symbol}_"
            backtest_files = [f for f in os.listdir('.') if f.startswith(backtest_pattern) and f.endswith('.csv')]

            if not backtest_files:
                # No historical data available
                self._historical_cache[cache_key] = 70.0  # Slightly positive default
                return 70.0

            # Use the most recent backtest file
            latest_file = sorted(backtest_files)[-1]
            df_backtest = pd.read_csv(latest_file)

            # Filter for similar signals
            signal_filter = df_backtest['signal'].str.contains(signal_direction.split()[0], case=False, na=False)
            similar_trades = df_backtest[signal_filter]

            if len(similar_trades) < 5:
                # Not enough historical data
                self._historical_cache[cache_key] = 70.0
                return 70.0

            # Calculate win rate
            if 'pnl' in similar_trades.columns:
                wins = (similar_trades['pnl'] > 0).sum()
                total = len(similar_trades)
                win_rate = wins / total

                # Convert win rate to confidence score
                # 70% win rate = 100 confidence, 50% = 70 confidence, 30% = 40 confidence
                confidence = max(0, min(100, 70 + (win_rate - 0.5) * 60))

            else:
                confidence = 70.0  # No PnL data available

            self._historical_cache[cache_key] = confidence
            return confidence

        except Exception:
            # Error loading historical data
            self._historical_cache[cache_key] = 70.0
            return 70.0

    def visualize_confidence_distribution(self, confidence_scores: List[float],
                                        symbol: str = '', save_path: str = None) -> None:
        """
        Create a histogram showing confidence score distribution.

        Args:
            confidence_scores: List of confidence scores (0-100)
            symbol: Optional symbol for the title
            save_path: Optional path to save the plot
        """
        if not confidence_scores:
            print("No confidence scores to visualize")
            return

        plt.figure(figsize=(10, 6))

        # Create histogram
        bins = np.arange(0, 101, 10)  # 0-10, 10-20, ..., 90-100
        plt.hist(confidence_scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')

        # Add statistics
        mean_conf = np.mean(confidence_scores)
        median_conf = np.median(confidence_scores)

        plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.1f}')
        plt.axvline(median_conf, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_conf:.1f}')

        # Formatting
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Score Distribution{" - " + symbol if symbol else ""}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)

        # Add confidence level zones
        plt.axvspan(0, 30, alpha=0.1, color='red', label='Low Confidence')
        plt.axvspan(30, 70, alpha=0.1, color='yellow', label='Medium Confidence')
        plt.axvspan(70, 100, alpha=0.1, color='green', label='High Confidence')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution plot saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def format_confidence_output(self, breakdown: ConfidenceBreakdown, symbol: str, signal: str) -> str:
        """
        Format confidence breakdown for display.

        Returns a formatted string showing the confidence score and breakdown.
        """
        return f"""
Confidence Analysis for {symbol} - {signal}
==========================================
Overall Confidence: {breakdown.overall}/100

Component Breakdown:
  • Technical Indicators: {breakdown.technical}/100 ({self.weights['technical']*100:.0f}% weight)
  • Sentiment Alignment:  {breakdown.sentiment}/100 ({self.weights['sentiment']*100:.0f}% weight)
  • Signal Strength:      {breakdown.strength}/100 ({self.weights['strength']*100:.0f}% weight)
  • Historical Reliability: {breakdown.historical}/100 ({self.weights['historical']*100:.0f}% weight)

Confidence Level: {'HIGH' if breakdown.overall >= 70 else 'MEDIUM' if breakdown.overall >= 40 else 'LOW'}
"""


def filter_signals_by_confidence(signals_df: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """
    Filter trading signals by minimum confidence threshold.

    Args:
        signals_df: DataFrame with trading signals and confidence scores
        min_confidence: Minimum confidence score (0-100) to include

    Returns:
        Filtered DataFrame containing only signals above the confidence threshold
    """
    if 'confidence' not in signals_df.columns:
        print("Warning: No confidence scores found in signals DataFrame")
        return signals_df

    filtered = signals_df[signals_df['confidence'] >= min_confidence]

    original_count = len(signals_df)
    filtered_count = len(filtered)

    print(f"Confidence filtering: {filtered_count}/{original_count} signals passed "
          f"(min confidence: {min_confidence})")

    return filtered


if __name__ == "__main__":
    # Example usage and testing
    print("CometAgent Confidence Scorer - Test Mode")

    # Create test data
    test_indicators = {
        'close': 150.0,
        'rsi14': 25.0,  # Oversold
        'macd': 0.5,
        'macd_signal': 0.3,
        'sma50': 145.0,
        'sma200': 140.0,
        'bb_up': 155.0,
        'bb_dn': 145.0,
        'bb_mid': 150.0
    }

    test_sentiment = {
        'sentiment_score': 0.3,    # Positive sentiment
        'x_only_score': 0.4,
        'reddit_only_score': 0.2
    }

    # Test confidence calculation
    scorer = ConfidenceScorer()
    confidence = scorer.calculate_confidence(
        test_indicators,
        test_sentiment,
        'Buy',
        'AAPL'
    )

    # Display results
    print(scorer.format_confidence_output(confidence, 'AAPL', 'Buy'))

    # Test visualization with sample data
    sample_scores = [45, 67, 78, 56, 89, 34, 72, 83, 91, 58, 76, 43, 88, 65, 79]
    scorer.visualize_confidence_distribution(sample_scores, 'Test Portfolio')