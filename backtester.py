#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtester.py
Backtest trading signals from agentic-trader.py against historical data.

Usage:
python backtester.py --symbol BTC-USD --start 2024-01-01 --end 2024-12-31 --horizon short

Features:
- Reuses signal generation logic from agentic-trader.py
- Simulates trades at regular intervals (weekly by default)
- Tracks portfolio performance vs buy-and-hold
- Calculates key performance metrics
- Generates matplotlib visualization
"""

import argparse
import datetime as dt
import os
import sys
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Handle potential version compatibility issues
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages: pip install numpy pandas matplotlib yfinance")
    sys.exit(1)

# Import and define signal generation functions
# We'll implement simplified versions to avoid complex import issues

def _coerce_symbol(portfolio_type: str, s: str) -> str:
    s = s.strip().upper()
    if portfolio_type == "crypto":
        if "-" not in s and "/" not in s:
            return f"{s}-USD"
    return s

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _simple_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    rsi14 = _rsi(close, 14)

    last = -1
    return {
        'close': float(close.iloc[last]),
        'sma50': float(sma50.iloc[last]),
        'sma200': float(sma200.iloc[last]) if not np.isnan(sma200.iloc[last]) else float('nan'),
        'ema12': float(ema12.iloc[last]),
        'ema26': float(ema26.iloc[last]),
        'macd': float(macd.iloc[last]),
        'macd_signal': float(macd_signal.iloc[last]),
        'rsi14': float(rsi14.iloc[last])
    }

def _simple_ta_score(ind: dict, horizon: str) -> float:
    """Simplified technical analysis scoring"""
    score = 0.0

    # Trend bias
    if not np.isnan(ind['sma200']):
        if ind['close'] > ind['sma200']:
            score += 0.5
        else:
            score -= 0.5
    else:
        if ind['close'] > ind['sma50']:
            score += 0.2
        else:
            score -= 0.2

    # Momentum via MACD
    if ind['macd'] > ind['macd_signal']:
        score += 0.25
    else:
        score -= 0.25

    # RSI (contrarian)
    rsi_term = ((50.0 - ind['rsi14']) / 50.0) * 0.2
    score += rsi_term

    # Horizon weighting
    if horizon == "short":
        score *= 1.00
    elif horizon == "medium":
        score *= 0.95
    else:
        score *= 0.90

    return max(min(score, 1.0), -1.0)

def _simple_decide(score: float) -> dict:
    """Make trading decision based on score"""
    if score >= 0.60:
        return {"action": "Strong Buy", "target_pct_delta": 3.0, "rationale": f"Score={score:+.2f} (high conviction)"}
    elif score >= 0.30:
        return {"action": "Buy", "target_pct_delta": 1.5, "rationale": f"Score={score:+.2f}"}
    elif score <= -0.60:
        return {"action": "Sell", "target_pct_delta": -3.0, "rationale": f"Score={score:+.2f} (high conviction)"}
    elif score <= -0.30:
        return {"action": "Trim", "target_pct_delta": -1.5, "rationale": f"Score={score:+.2f}"}
    else:
        return {"action": "Hold", "target_pct_delta": 0.0, "rationale": f"Score={score:+.2f}"}

class BacktestPosition:
    def __init__(self, cash: float = 10000.0):
        self.cash = cash
        self.shares = 0.0
        self.total_value = cash
        self.trades: List[Dict] = []

    def get_portfolio_value(self, current_price: float) -> float:
        return self.cash + (self.shares * current_price)

    def execute_trade(self, date: pd.Timestamp, price: float, action: str,
                     target_pct_delta: float, current_value: float) -> Dict:
        """Execute a trade based on signal"""

        # Calculate target position change
        if action in ["Strong Buy", "Buy"]:
            # Buying: convert target_pct_delta to dollar amount to invest
            target_investment = current_value * (target_pct_delta / 100)
            shares_to_buy = min(target_investment / price, self.cash / price)

            if shares_to_buy > 0.01:  # Minimum trade threshold
                cost = shares_to_buy * price
                self.cash -= cost
                self.shares += shares_to_buy

                trade = {
                    'date': date,
                    'action': action,
                    'shares': shares_to_buy,
                    'price': price,
                    'value': cost,
                    'cash_after': self.cash,
                    'shares_after': self.shares
                }
                self.trades.append(trade)
                return trade

        elif action in ["Sell", "Trim"]:
            # Selling: convert target_pct_delta to shares to sell
            target_reduction = current_value * (abs(target_pct_delta) / 100)
            shares_to_sell = min(target_reduction / price, self.shares)

            if shares_to_sell > 0.01:  # Minimum trade threshold
                proceeds = shares_to_sell * price
                self.cash += proceeds
                self.shares -= shares_to_sell

                trade = {
                    'date': date,
                    'action': action,
                    'shares': -shares_to_sell,
                    'price': price,
                    'value': -proceeds,
                    'cash_after': self.cash,
                    'shares_after': self.shares
                }
                self.trades.append(trade)
                return trade

        return None

class Backtester:
    def __init__(self, symbol: str, start_date: str, end_date: str,
                 horizon: str, initial_cash: float = 10000.0,
                 rebalance_freq: str = "W"):
        """
        Initialize backtester

        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "AAPL")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            horizon: Investment horizon ("short", "medium", "long")
            initial_cash: Starting portfolio value
            rebalance_freq: Rebalancing frequency ("D"=daily, "W"=weekly, "M"=monthly)
        """
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.horizon = horizon
        self.initial_cash = initial_cash
        self.rebalance_freq = rebalance_freq
        self.portfolio_type = "crypto" if "-" in symbol or symbol in ["BTC", "ETH"] else "stocks"

        # Results storage
        self.backtest_results: List[Dict] = []
        self.performance_metrics: Dict = {}

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch extended historical data for the backtest period"""
        # Get additional data before start date for technical indicators
        extended_start = self.start_date - pd.DateOffset(months=12)

        # Use a longer period to ensure we have enough data
        period_mapping = {
            (self.end_date - self.start_date).days: "max" if (self.end_date - self.start_date).days > 365 else "2y"
        }

        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)

            # Try to get data for the specific date range
            df = ticker.history(
                start=extended_start.strftime('%Y-%m-%d'),
                end=(self.end_date + pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
                interval='1d',
                auto_adjust=True
            )

            if df.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            return df

        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return pd.DataFrame()

    def generate_signal_at_date(self, data_up_to_date: pd.DataFrame, signal_date: pd.Timestamp) -> Dict:
        """Generate trading signal using simplified technical analysis"""
        try:
            # Ensure we have enough data for indicators
            if len(data_up_to_date) < 50:
                return {
                    'date': signal_date,
                    'action': 'Hold',
                    'target_pct_delta': 0.0,
                    'total_score': 0.0,
                    'ta_score': 0.0,
                    'sentiment_score': 0.0,
                    'close_price': data_up_to_date['Close'].iloc[-1] if not data_up_to_date.empty else np.nan
                }

            # Calculate indicators using simplified function
            indicators = _simple_indicators(data_up_to_date)

            # Get technical analysis score
            ta_score = _simple_ta_score(indicators, self.horizon)

            # For backtesting, we'll use only technical analysis
            # Sentiment data is not available historically
            total_score = ta_score  # Pure technical analysis

            # Make decision
            decision = _simple_decide(total_score)

            return {
                'date': signal_date,
                'action': decision['action'],
                'target_pct_delta': decision['target_pct_delta'],
                'total_score': total_score,
                'ta_score': ta_score,
                'sentiment_score': 0.0,  # Not available for historical data
                'close_price': indicators['close'],
                'rsi': indicators['rsi14'],
                'macd': indicators['macd'],
                'reasoning': decision['rationale']
            }

        except Exception as e:
            print(f"Error generating signal for {signal_date}: {e}")
            return {
                'date': signal_date,
                'action': 'Hold',
                'target_pct_delta': 0.0,
                'total_score': 0.0,
                'ta_score': 0.0,
                'sentiment_score': 0.0,
                'close_price': np.nan
            }

    def run_backtest(self) -> pd.DataFrame:
        """Run the complete backtest"""
        print(f"Running backtest for {self.symbol} from {self.start_date.date()} to {self.end_date.date()}")
        print(f"Horizon: {self.horizon}, Rebalance frequency: {self.rebalance_freq}")

        # Fetch historical data
        historical_data = self.fetch_historical_data()
        if historical_data.empty:
            raise ValueError("Could not fetch historical data")

        print(f"Loaded {len(historical_data)} days of historical data")

        # Initialize portfolio
        portfolio = BacktestPosition(self.initial_cash)

        # Generate rebalancing dates
        rebalance_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self.rebalance_freq
        )

        print(f"Generated {len(rebalance_dates)} rebalancing dates")

        results = []

        for i, date in enumerate(rebalance_dates):
            # Handle timezone-aware vs timezone-naive comparison
            if historical_data.index.tz is not None:
                if date.tz is None:
                    date = date.tz_localize(historical_data.index.tz)

            # Get data up to this date for signal generation
            data_up_to_date = historical_data[historical_data.index <= date]

            if data_up_to_date.empty:
                continue

            # Generate signal
            signal = self.generate_signal_at_date(data_up_to_date, date)
            current_price = signal['close_price']

            if np.isnan(current_price):
                continue

            # Calculate current portfolio value
            portfolio_value = portfolio.get_portfolio_value(current_price)

            # Execute trade if signal indicates action
            trade_executed = None
            if signal['action'] != 'Hold':
                trade_executed = portfolio.execute_trade(
                    date, current_price, signal['action'],
                    signal['target_pct_delta'], portfolio_value
                )

            # Calculate buy-and-hold value for comparison
            if i == 0:
                # Initialize buy-and-hold position
                bh_shares = self.initial_cash / current_price
                bh_value = self.initial_cash
            else:
                bh_value = bh_shares * current_price

            # Store results
            result = {
                'date': date,
                'price': current_price,
                'signal_action': signal['action'],
                'target_pct_delta': signal['target_pct_delta'],
                'total_score': signal['total_score'],
                'ta_score': signal['ta_score'],
                'sentiment_score': signal['sentiment_score'],
                'portfolio_value': portfolio_value,
                'cash': portfolio.cash,
                'shares': portfolio.shares,
                'buy_hold_value': bh_value,
                'trade_executed': trade_executed is not None,
                'rsi': signal.get('rsi', np.nan),
                'macd': signal.get('macd', np.nan)
            }
            results.append(result)

            if i % 10 == 0:  # Progress update
                print(f"Processed {i+1}/{len(rebalance_dates)} dates...")

        self.backtest_results = results

        # Convert to DataFrame for easier analysis
        df_results = pd.DataFrame(results)
        df_results.set_index('date', inplace=True)

        return df_results

    def calculate_metrics(self, df_results: pd.DataFrame) -> Dict:
        """Calculate key performance metrics"""
        if df_results.empty:
            return {}

        # Final values
        final_portfolio_value = df_results['portfolio_value'].iloc[-1]
        final_bh_value = df_results['buy_hold_value'].iloc[-1]

        # Returns
        total_return = (final_portfolio_value / self.initial_cash) - 1
        bh_return = (final_bh_value / self.initial_cash) - 1
        excess_return = total_return - bh_return

        # Trading statistics
        trades = [t for t in self.backtest_results if t['trade_executed']]
        total_trades = len(trades)

        # Win rate calculation
        winning_trades = 0
        total_gains = 0
        total_losses = 0
        trade_returns = []

        if total_trades > 0:
            for i, trade_result in enumerate(df_results[df_results['trade_executed']].iterrows()):
                date, row = trade_result
                # Look ahead to next rebalance to see if trade was profitable
                future_rows = df_results[df_results.index > date]
                if not future_rows.empty:
                    next_price = future_rows['price'].iloc[0] if len(future_rows) > 0 else row['price']
                    current_price = row['price']

                    if row['signal_action'] in ['Buy', 'Strong Buy']:
                        trade_return = (next_price / current_price) - 1
                    else:  # Sell, Trim
                        trade_return = (current_price / next_price) - 1

                    trade_returns.append(trade_return)

                    if trade_return > 0:
                        winning_trades += 1
                        total_gains += trade_return
                    else:
                        total_losses += abs(trade_return)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_gain = (total_gains / winning_trades) if winning_trades > 0 else 0
        avg_loss = (total_losses / (total_trades - winning_trades)) if (total_trades - winning_trades) > 0 else 0

        # Drawdown calculation
        portfolio_values = df_results['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Sharpe ratio (simplified - using daily returns)
        portfolio_returns = portfolio_values.pct_change().dropna()
        if len(portfolio_returns) > 1:
            sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(252) if len(portfolio_returns) > 1 else 0

        metrics = {
            'total_return': total_return * 100,
            'buy_hold_return': bh_return * 100,
            'excess_return': excess_return * 100,
            'final_portfolio_value': final_portfolio_value,
            'final_bh_value': final_bh_value,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_gain': avg_gain * 100,
            'avg_loss': avg_loss * 100,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility * 100
        }

        self.performance_metrics = metrics
        return metrics

    def generate_chart(self, df_results: pd.DataFrame, save_path: Optional[str] = None):
        """Generate performance visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Backtest Results: {self.symbol} ({self.start_date.date()} to {self.end_date.date()})',
                    fontsize=16, fontweight='bold')

        # Portfolio value vs Buy & Hold
        axes[0,0].plot(df_results.index, df_results['portfolio_value'],
                      label='Strategy', linewidth=2, color='blue')
        axes[0,0].plot(df_results.index, df_results['buy_hold_value'],
                      label='Buy & Hold', linewidth=2, color='orange')
        axes[0,0].set_title('Portfolio Value Over Time')
        axes[0,0].set_ylabel('Value ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Price with trade signals
        axes[0,1].plot(df_results.index, df_results['price'],
                      label='Price', linewidth=1, color='black')

        # Mark buy/sell signals
        buy_signals = df_results[df_results['signal_action'].isin(['Buy', 'Strong Buy'])]
        sell_signals = df_results[df_results['signal_action'].isin(['Sell', 'Trim'])]

        axes[0,1].scatter(buy_signals.index, buy_signals['price'],
                         color='green', marker='^', s=50, label='Buy Signal')
        axes[0,1].scatter(sell_signals.index, sell_signals['price'],
                         color='red', marker='v', s=50, label='Sell Signal')

        axes[0,1].set_title(f'{self.symbol} Price with Signals')
        axes[0,1].set_ylabel('Price ($)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Cumulative returns
        strategy_returns = (df_results['portfolio_value'] / self.initial_cash - 1) * 100
        bh_returns = (df_results['buy_hold_value'] / self.initial_cash - 1) * 100

        axes[1,0].plot(df_results.index, strategy_returns,
                      label='Strategy', linewidth=2, color='blue')
        axes[1,0].plot(df_results.index, bh_returns,
                      label='Buy & Hold', linewidth=2, color='orange')
        axes[1,0].set_title('Cumulative Returns (%)')
        axes[1,0].set_ylabel('Return (%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Drawdown
        running_max = df_results['portfolio_value'].expanding().max()
        drawdown = (df_results['portfolio_value'] - running_max) / running_max * 100

        axes[1,1].fill_between(df_results.index, drawdown, 0,
                              color='red', alpha=0.3)
        axes[1,1].plot(df_results.index, drawdown, color='red')
        axes[1,1].set_title('Portfolio Drawdown (%)')
        axes[1,1].set_ylabel('Drawdown (%)')
        axes[1,1].grid(True, alpha=0.3)

        # Signal scores over time
        axes[2,0].plot(df_results.index, df_results['total_score'],
                      label='Total Score', linewidth=2)
        axes[2,0].plot(df_results.index, df_results['ta_score'],
                      label='Technical', alpha=0.7)
        axes[2,0].plot(df_results.index, df_results['sentiment_score'],
                      label='Sentiment', alpha=0.7)
        axes[2,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2,0].set_title('Signal Scores Over Time')
        axes[2,0].set_ylabel('Score')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)

        # Performance metrics summary
        metrics_text = f"""Performance Summary:
Total Return: {self.performance_metrics.get('total_return', 0):.1f}%
Buy & Hold Return: {self.performance_metrics.get('buy_hold_return', 0):.1f}%
Excess Return: {self.performance_metrics.get('excess_return', 0):.1f}%

Total Trades: {self.performance_metrics.get('total_trades', 0)}
Win Rate: {self.performance_metrics.get('win_rate', 0):.1f}%
Avg Gain: {self.performance_metrics.get('avg_gain', 0):.1f}%
Avg Loss: {self.performance_metrics.get('avg_loss', 0):.1f}%

Max Drawdown: {self.performance_metrics.get('max_drawdown', 0):.1f}%
Sharpe Ratio: {self.performance_metrics.get('sharpe_ratio', 0):.2f}
Volatility: {self.performance_metrics.get('volatility', 0):.1f}%"""

        axes[2,1].text(0.05, 0.95, metrics_text, transform=axes[2,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        axes[2,1].set_xlim(0, 1)
        axes[2,1].set_ylim(0, 1)
        axes[2,1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")

        # plt.show()  # Don't show interactively, just save

def main():
    parser = argparse.ArgumentParser(description="Backtest trading signals from agentic-trader.py")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC-USD, AAPL)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--horizon", required=True, choices=["short", "medium", "long"],
                       help="Investment horizon")
    parser.add_argument("--initial-cash", type=float, default=10000.0,
                       help="Initial portfolio value (default: 10000)")
    parser.add_argument("--freq", default="W", choices=["D", "W", "M"],
                       help="Rebalancing frequency: D=daily, W=weekly, M=monthly (default: W)")
    parser.add_argument("--save-chart", help="Path to save the performance chart")

    args = parser.parse_args()

    try:
        # Coerce symbol to proper format
        portfolio_type = "crypto" if "-" in args.symbol or args.symbol in ["BTC", "ETH"] else "stocks"
        symbol = _coerce_symbol(portfolio_type, args.symbol)

        # Create backtester
        backtester = Backtester(
            symbol=symbol,
            start_date=args.start,
            end_date=args.end,
            horizon=args.horizon,
            initial_cash=args.initial_cash,
            rebalance_freq=args.freq
        )

        # Run backtest
        print(f"\n{'='*60}")
        print(f"BACKTESTING {symbol.upper()}")
        print(f"{'='*60}")

        df_results = backtester.run_backtest()

        if df_results.empty:
            print("No results generated. Check symbol and date range.")
            return

        # Calculate metrics
        metrics = backtester.calculate_metrics(df_results)

        # Print results
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Period: {args.start} to {args.end}")
        print(f"Horizon: {args.horizon}")
        print(f"Initial Capital: ${args.initial_cash:,.2f}")
        print(f"Rebalancing: {args.freq}")
        print(f"\n{'Portfolio Performance:':<25}")
        print(f"{'Final Value:':<25} ${metrics.get('final_portfolio_value', 0):,.2f}")
        print(f"{'Total Return:':<25} {metrics.get('total_return', 0):+.2f}%")
        print(f"\n{'Buy & Hold Comparison:':<25}")
        print(f"{'B&H Final Value:':<25} ${metrics.get('final_bh_value', 0):,.2f}")
        print(f"{'B&H Return:':<25} {metrics.get('buy_hold_return', 0):+.2f}%")
        print(f"{'Excess Return:':<25} {metrics.get('excess_return', 0):+.2f}%")
        print(f"\n{'Trading Statistics:':<25}")
        print(f"{'Total Trades:':<25} {metrics.get('total_trades', 0)}")
        print(f"{'Win Rate:':<25} {metrics.get('win_rate', 0):.1f}%")
        print(f"{'Average Gain:':<25} {metrics.get('avg_gain', 0):+.2f}%")
        print(f"{'Average Loss:':<25} {metrics.get('avg_loss', 0):-.2f}%")
        print(f"\n{'Risk Metrics:':<25}")
        print(f"{'Max Drawdown:':<25} {metrics.get('max_drawdown', 0):.2f}%")
        print(f"{'Sharpe Ratio:':<25} {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"{'Volatility:':<25} {metrics.get('volatility', 0):.2f}%")

        # Generate visualization
        print(f"\nGenerating performance chart...")
        chart_path = args.save_chart or f"backtest_{symbol.replace('-', '_')}_{args.start}_{args.end}.png"
        backtester.generate_chart(df_results, chart_path)

        # Save detailed results to CSV
        results_path = f"backtest_results_{symbol.replace('-', '_')}_{args.start}_{args.end}.csv"
        df_results.to_csv(results_path)
        print(f"Detailed results saved to: {results_path}")

        print(f"\n{'='*60}")
        print("Backtest completed successfully!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()