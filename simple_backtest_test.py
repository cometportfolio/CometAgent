#!/usr/bin/env python3
"""
Simple test of backtesting functionality without sentiment analysis
"""

import argparse
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def simple_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def simple_signal(df: pd.DataFrame) -> dict:
    """Generate a simple signal based on RSI and moving averages"""
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    rsi = simple_rsi(close)

    latest_close = close.iloc[-1]
    latest_sma50 = sma50.iloc[-1]
    latest_rsi = rsi.iloc[-1]

    score = 0.0

    # Trend signal
    if latest_close > latest_sma50:
        score += 0.5
    else:
        score -= 0.5

    # RSI signal (contrarian)
    if latest_rsi < 30:
        score += 0.3
    elif latest_rsi > 70:
        score -= 0.3

    # Generate action
    if score >= 0.4:
        action = "Buy"
        target_pct = 2.0
    elif score <= -0.4:
        action = "Sell"
        target_pct = -2.0
    else:
        action = "Hold"
        target_pct = 0.0

    return {
        'action': action,
        'target_pct_delta': target_pct,
        'score': score,
        'close': latest_close,
        'rsi': latest_rsi
    }

def run_simple_backtest(symbol, start_date, end_date):
    """Run a simple backtest"""

    print(f"Running simple backtest for {symbol}")

    # Fetch data
    ticker = yf.Ticker(symbol)
    extended_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)

    df = ticker.history(
        start=extended_start.strftime('%Y-%m-%d'),
        end=(pd.to_datetime(end_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
        interval='1d',
        auto_adjust=True
    )

    if df.empty:
        print("No data found")
        return

    print(f"Loaded {len(df)} days of data")

    # Generate weekly rebalance dates
    rebalance_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq='W'
    )

    # Initialize portfolio
    initial_cash = 10000.0
    cash = initial_cash
    shares = 0.0

    results = []
    bh_shares = initial_cash / df.loc[df.index >= start_date]['Close'].iloc[0]

    for date in rebalance_dates:
        # Handle timezone-aware vs timezone-naive comparison
        if df.index.tz is not None:
            date = date.tz_localize(df.index.tz)
        data_up_to_date = df[df.index <= date]
        if len(data_up_to_date) < 60:
            continue

        signal = simple_signal(data_up_to_date)
        current_price = signal['close']
        portfolio_value = cash + shares * current_price
        bh_value = bh_shares * current_price

        # Execute trade
        if signal['action'] == 'Buy' and signal['target_pct_delta'] > 0:
            target_investment = portfolio_value * (signal['target_pct_delta'] / 100)
            shares_to_buy = min(target_investment / current_price, cash / current_price)
            if shares_to_buy > 0.01:
                cash -= shares_to_buy * current_price
                shares += shares_to_buy

        elif signal['action'] == 'Sell' and signal['target_pct_delta'] < 0:
            target_reduction = portfolio_value * (abs(signal['target_pct_delta']) / 100)
            shares_to_sell = min(target_reduction / current_price, shares)
            if shares_to_sell > 0.01:
                cash += shares_to_sell * current_price
                shares -= shares_to_sell

        portfolio_value = cash + shares * current_price

        results.append({
            'date': date,
            'price': current_price,
            'action': signal['action'],
            'score': signal['score'],
            'rsi': signal['rsi'],
            'portfolio_value': portfolio_value,
            'bh_value': bh_value,
            'cash': cash,
            'shares': shares
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Calculate metrics
    final_portfolio = df_results['portfolio_value'].iloc[-1]
    final_bh = df_results['bh_value'].iloc[-1]

    total_return = (final_portfolio / initial_cash - 1) * 100
    bh_return = (final_bh / initial_cash - 1) * 100
    excess_return = total_return - bh_return

    print(f"\n{'='*50}")
    print(f"SIMPLE BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Portfolio: ${final_portfolio:,.2f}")
    print(f"Final B&H Value: ${final_bh:,.2f}")
    print(f"Strategy Return: {total_return:+.2f}%")
    print(f"B&H Return: {bh_return:+.2f}%")
    print(f"Excess Return: {excess_return:+.2f}%")

    # Simple plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_results['date'], df_results['portfolio_value'], label='Strategy', linewidth=2)
    plt.plot(df_results['date'], df_results['bh_value'], label='Buy & Hold', linewidth=2)
    plt.title(f'{symbol} Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save chart
    chart_path = f"simple_backtest_{symbol.replace('-', '_')}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")

    plt.show()

    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple backtest")
    parser.add_argument("--symbol", default="BTC-USD", help="Symbol to test")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2024-03-31", help="End date")

    args = parser.parse_args()

    results = run_simple_backtest(args.symbol, args.start, args.end)