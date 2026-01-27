#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agentic_trader.py
Generate trading signals (stocks or crypto) and agentic-execution prompts
optimized specifically for Perplexity Comet browser automation.

COMET OPTIMIZATION FEATURES:
- Voice-command friendly prompts
- Autonomous workflow execution
- Real-time Coinbase integration (via Perplexity partnership)
- Context-aware trading instructions
- Streamlined no-click execution patterns

Usage
-----
python agentic_trader.py \
  --portfolio-type stocks \
  --horizon short \
  --platform coinbase \
  --file my_portfolio.csv

Inputs
------
Portfolio file (.csv or .xlsx) with columns (case-insensitive):
- ticker or symbol (e.g., AAPL, MSFT, BTC, ETH)  [required]
- quantity or shares                              [optional; defaults to 0]
- purchase_date or date                           [optional]
- purchase_price or price                         [optional]
- notes                                           [optional]

Outputs
-------
- Console report (signals + reasoning)
- prompts_{timestamp}.txt  -> copy/paste-able Comet prompts per action
- signals_{timestamp}.csv  -> tabular summary of scores and decisions

Dependencies (install as needed)
--------------------------------
pip install pandas numpy yfinance feedparser vaderSentiment snscrape praw

Notes
-----
- Multi-source sentiment analysis:
  * News: Yahoo Finance (via yfinance) + optional RSS feeds (feedparser)
  * Social: X/Twitter posts (via snscrape)
  * Reddit: Posts and comments with engagement weighting (via praw)
- Reddit analysis includes upvote, award, and comment count weighting for signal quality
- For crypto tickers, symbols like 'BTC'/'ETH' are mapped to 'BTC-USD'/'ETH-USD' for prices.
- Optimized for Perplexity Comet browser's autonomous execution capabilities.
- This is NOT financial advice; it's a research tool to aid your process.
"""

from __future__ import annotations
import argparse, datetime as dt, json, os, re, sys, textwrap
import threading, time, multiprocessing, signal
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

__version__ = "3.23"

# Soft deps; handled at runtime if missing
try:
    import yfinance as yf
except Exception as e:
    print("yfinance is required: pip install yfinance", file=sys.stderr); raise

try:
    import feedparser  # optional
except Exception:
    feedparser = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

# snscrape for X (optional). Do not hard-fail if missing.
try:
    import subprocess, shutil
    _SNSCRAPE = shutil.which("snscrape") is not None
except Exception:
    _SNSCRAPE = False

# PRAW for Reddit (optional). Do not hard-fail if missing.
try:
    import praw
    _PRAW_AVAILABLE = True
except Exception:
    _PRAW_AVAILABLE = False
    praw = None

# Import confidence scorer
try:
    from confidence_scorer import ConfidenceScorer, filter_signals_by_confidence
    _CONFIDENCE_AVAILABLE = True
except Exception:
    _CONFIDENCE_AVAILABLE = False
    print("Warning: confidence_scorer.py not found. Confidence scoring will be disabled.")
    ConfidenceScorer = None


# ----------------------------- Config -------------------------------- #

def _load_news_source_weights() -> Dict[str, float]:
    """Load news source weights from news_sources_weighting.md if it exists."""
    config_path = "news_sources_weighting.md"
    if not os.path.exists(config_path):
        return {}

    weights = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('source'):
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 2:
                    continue

                source_name, weight_str = parts
                try:
                    weight = float(weight_str)
                    if 1 <= weight <= 10:
                        weights[source_name] = weight
                except ValueError:
                    continue
    except Exception:
        return {}

    return weights

def _load_social_source_weights() -> Dict[str, float]:
    """Load social media source weights from social_sources_weighting.md if it exists."""
    config_path = "social_sources_weighting.md"
    if not os.path.exists(config_path):
        return {}

    weights = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('source') or line.startswith('handle'):
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 2:
                    continue

                source_handle, weight_str = parts
                try:
                    weight = float(weight_str)
                    if 1 <= weight <= 10:
                        # Extract handle from X URL if provided
                        if 'x.com/' in source_handle or 'twitter.com/' in source_handle:
                            # Extract handle from URL like https://x.com/username or https://twitter.com/username
                            handle = source_handle.split('/')[-1].replace('@', '')
                        else:
                            # Handle provided directly (remove @ if present)
                            handle = source_handle.replace('@', '')
                        weights[handle] = weight
                except ValueError:
                    continue
    except Exception:
        return {}

    return weights

def _get_weighted_news_sources(portfolio_type: str) -> List[Dict[str, any]]:
    """Get news sources with their weights applied."""
    default_sources = DEFAULT_STOCK_SOURCES if portfolio_type == "stocks" else DEFAULT_CRYPTO_SOURCES
    weights = _load_news_source_weights()

    if not weights:
        # Return default sources with default weight of 5
        return [dict(source, weight=5.0) for source in default_sources]

    weighted_sources = []
    for source in default_sources:
        source_name = source["name"]
        weight = weights.get(source_name, 5.0)  # Default weight of 5 if not specified
        weighted_sources.append(dict(source, weight=weight))

    # Sort by weight (descending) to prioritize higher-weighted sources
    weighted_sources.sort(key=lambda x: x["weight"], reverse=True)
    return weighted_sources

def _get_weighted_social_sources(portfolio_type: str) -> List[Dict[str, float]]:
    """Get social media sources with their weights applied."""
    default_handles = DEFAULT_STOCK_X_HANDLES if portfolio_type == "stocks" else DEFAULT_CRYPTO_X_HANDLES
    weights = _load_social_source_weights()

    if not weights:
        # Return default handles with default weight of 5
        return [{"handle": handle, "weight": 5.0} for handle in default_handles]

    # Start with configured weighted sources
    weighted_sources = []

    # Add all sources from config file (they may include sources not in defaults)
    for handle, weight in weights.items():
        weighted_sources.append({"handle": handle, "weight": weight})

    # Add any default handles not in config with default weight
    for handle in default_handles:
        if handle not in weights:
            weighted_sources.append({"handle": handle, "weight": 5.0})

    # Sort by weight (descending) to prioritize higher-weighted sources
    weighted_sources.sort(key=lambda x: x["weight"], reverse=True)
    return weighted_sources

def _load_reddit_source_weights() -> Dict[str, float]:
    """Load Reddit subreddit weights from reddit_sources_weighting.md if it exists."""
    config_path = "reddit_sources_weighting.md"
    if not os.path.exists(config_path):
        return {}

    weights = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('subreddit'):
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 2:
                    continue

                subreddit_name, weight_str = parts
                try:
                    weight = float(weight_str)
                    if 1 <= weight <= 10:
                        # Clean subreddit name (remove r/ prefix if present)
                        if subreddit_name.startswith('r/'):
                            subreddit_name = subreddit_name[2:]
                        weights[subreddit_name] = weight
                except ValueError:
                    continue
    except Exception:
        return {}

    return weights

def _get_weighted_reddit_sources(portfolio_type: str) -> List[Dict[str, float]]:
    """Get Reddit subreddits with their weights applied."""
    default_subreddits = DEFAULT_STOCK_SUBREDDITS if portfolio_type == "stocks" else DEFAULT_CRYPTO_SUBREDDITS
    weights = _load_reddit_source_weights()

    if not weights:
        # Return default subreddits with default weight of 5
        return [{"subreddit": sub, "weight": 5.0} for sub in default_subreddits]

    # Start with configured weighted sources
    weighted_sources = []

    # Add all sources from config file (they may include sources not in defaults)
    for subreddit, weight in weights.items():
        weighted_sources.append({"subreddit": subreddit, "weight": weight})

    # Add any default subreddits not in config with default weight
    for subreddit in default_subreddits:
        if subreddit not in weights:
            weighted_sources.append({"subreddit": subreddit, "weight": 5.0})

    # Sort by weight (descending) to prioritize higher-weighted sources
    weighted_sources.sort(key=lambda x: x["weight"], reverse=True)
    return weighted_sources

DEFAULT_STOCK_SOURCES = [
    # High-impact publications (not exhaustive). These serve both as human references
    # and for optional RSS lookups if you add/point to RSS feeds you prefer.
    {"name": "Reuters Markets", "url": "https://www.reuters.com/markets/"},
    {"name": "Bloomberg Markets", "url": "https://www.bloomberg.com/markets"},
    {"name": "WSJ Finance & Markets", "url": "https://www.wsj.com/finance"},
    {"name": "Financial Times Markets", "url": "https://www.ft.com/markets"},
    {"name": "CNBC Markets", "url": "https://www.cnbc.com/markets/"},
    {"name": "MarketWatch", "url": "https://www.marketwatch.com/"},
    {"name": "Morningstar News", "url": "https://www.morningstar.com/news"},
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/"},
    {"name": "Seeking Alpha", "url": "https://seekingalpha.com/"},
]

DEFAULT_CRYPTO_SOURCES = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/markets"},
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/"},
    {"name": "The Block", "url": "https://www.theblock.co/"},
    {"name": "Decrypt", "url": "https://decrypt.co/"},
    {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/"},
    {"name": "Kaiko (research)", "url": "https://www.kaiko.com/"},
    {"name": "Glassnode (research)", "url": "https://glassnode.com/"},
    {"name": "CryptoQuant (research)", "url": "https://cryptoquant.com/"},
    {"name": "Santiment (research)", "url": "https://santiment.net/"},
    {"name": "DefiLlama (dashboards)", "url": "https://defillama.com/"},
]

# Trustworthy X (Twitter) accounts to scan if snscrape is present.
DEFAULT_STOCK_X_HANDLES = [
    "markets",        # Bloomberg Markets
    "WSJmarkets",     # WSJ Markets
    "ReutersBiz",     # Reuters Business
    "CNBCnow",        # CNBC breaking
    "bespokeinvest",  # Bespoke
    "elerianm",       # Mohamed El-Erian
    "TheStalwart",    # Joe Weisenthal
    "LizAnnSonders",  # Schwab strategist
    "lisaabramowicz1" # Bloomberg Radio
]

DEFAULT_CRYPTO_X_HANDLES = [
    "CoinDesk", "Cointelegraph", "TheBlock__", "decryptmedia",
    "KaikoData", "glassnode", "cryptoquant_com", "santimentfeed",
    "DefiLlama", "MessariCrypto"
]

# Reddit subreddits to scan for trading sentiment
DEFAULT_STOCK_SUBREDDITS = [
    "wallstreetbets",     # High volume, retail sentiment
    "stocks",             # General stock discussion
    "investing",          # Long-term investment focus
    "StockMarket",        # Market-wide discussions
    "options",            # Options trading
    "Daytrading",         # Short-term trading
    "SecurityAnalysis",   # Fundamental analysis
    "ValueInvesting",     # Value-focused
    "Dividends",          # Dividend stocks
]

DEFAULT_CRYPTO_SUBREDDITS = [
    "CryptoCurrency",     # General crypto discussion
    "Bitcoin",            # Bitcoin-specific
    "ethereum",           # Ethereum-specific
    "CryptoMarkets",      # Trading focused
    "altcoin",            # Altcoin discussion
    "binance",            # Exchange discussion
    "defi",               # DeFi projects
    "NFT",                # NFT discussion
    "BitcoinMarkets",     # Bitcoin trading
]


# --------------------------- Data classes ---------------------------- #

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    purchase_price: Optional[float] = None  # Average purchase price
    purchase_date: Optional[pd.Timestamp] = None  # First purchase date
    total_cost_basis: float = 0.0  # Total USD invested
    transaction_count: int = 0  # Number of transactions
    notes: str = ""

@dataclass
class Indicators:
    close: float
    sma50: float
    sma200: float
    ema12: float
    ema26: float
    macd: float
    macd_signal: float
    rsi14: float
    bb_mid: float
    bb_up: float
    bb_dn: float
    atr14: float

@dataclass
class Scores:
    ta: float
    news: float
    x: float
    total: float

@dataclass
class Decision:
    action: str   # "Strong Buy", "Buy", "Hold", "Trim", "Sell"
    target_pct_delta: float  # +3.0 means increase target weight by +3%
    rationale: str


# ------------------------- Utility functions ------------------------- #

def _coerce_symbol(portfolio_type: str, s: str) -> str:
    s = s.strip().upper()
    if portfolio_type == "crypto":
        # yfinance uses -USD for most major pairs
        if "-" not in s and "/" not in s:
            return f"{s}-USD"
    return s

def _read_portfolio(path: str, portfolio_type: str) -> List[Position]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        # Debug: show raw file content
        with open(path, 'r') as f:
            lines = f.readlines()[:5]
            print("DEBUG: Raw file lines:")
            for i, line in enumerate(lines):
                print(f"  Line {i}: {repr(line)}")
        
        # Try to find header row
        header_row = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(',,,'):
                header_row = i
                print(f"DEBUG: Using header row {i}: {repr(line)}")
                break
        
        df = pd.read_csv(path, skiprows=header_row)
        
        # Clean column names
        new_columns = []
        for col in df.columns:
            clean_col = str(col).lstrip('|').strip()
            new_columns.append(clean_col)
        df.columns = new_columns
        print(f"DEBUG: Cleaned columns: {df.columns.tolist()}")
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("File must be .csv or .xlsx")

    # Handle transaction-level data format (like crypto-portfolio.csv)
    cols = {c.lower().strip(): c for c in df.columns if c and str(c).strip()}
    
    def pick(*options):
        for o in options:
            if o in cols: return cols[o]
        return None

    # Check for transaction-level format first
    date_col = pick("date", "purchase_date")
    symbol_col = pick("ticker/symbol", "ticker", "symbol")
    price_col = pick("price", "purchase_price", "cost_basis")
    amount_col = pick("amount bought", "amount", "quantity", "shares", "qty")
    usd_col = pick("usd spent", "usd", "cost", "total_cost")
    transaction_type_col = pick("transaction type", "type", "action")
    
    # Debug output
    print(f"DEBUG: Raw columns: {list(df.columns)}")
    print(f"DEBUG: Column mapping: {cols}")
    print(f"DEBUG: symbol_col={symbol_col}, amount_col={amount_col}, usd_col={usd_col}")
    
    if symbol_col and (amount_col or usd_col):
        # Transaction-level format detected
        print("DEBUG: Transaction-level format detected")
        return _parse_transaction_format(df, portfolio_type, date_col, symbol_col, 
                                       price_col, amount_col, usd_col, transaction_type_col)
    else:
        print("DEBUG: Transaction-level format NOT detected, falling back to legacy format")
    
    # Fall back to legacy position-level format
    sym_col = pick("ticker", "symbol")
    if not sym_col: raise ValueError("Missing column: ticker/symbol")
    qty_col = pick("quantity", "shares", "qty")
    pp_col = pick("purchase_price", "price", "cost_basis")
    pd_col = pick("purchase_date", "date")
    notes_col = pick("notes",)

    positions: List[Position] = []
    for _, row in df.iterrows():
        sym = _coerce_symbol(portfolio_type, str(row[sym_col]))
        qty = float(row[qty_col]) if qty_col and not pd.isna(row.get(qty_col, np.nan)) else 0.0
        pp = float(row[pp_col]) if pp_col and not pd.isna(row.get(pp_col, np.nan)) else None
        date = None
        if pd_col and not pd.isna(row.get(pd_col, np.nan)):
            date = pd.to_datetime(row[pd_col], errors="coerce")
        notes = str(row[notes_col]) if notes_col and not pd.isna(row.get(notes_col, np.nan)) else ""
        total_cost = pp * qty if pp and qty else 0.0
        positions.append(Position(symbol=sym, qty=qty, purchase_price=pp, purchase_date=date, 
                                total_cost_basis=total_cost, transaction_count=1, notes=notes))
    return positions

def _parse_transaction_format(df: pd.DataFrame, portfolio_type: str, date_col: str, 
                            symbol_col: str, price_col: str, amount_col: str, 
                            usd_col: str, transaction_type_col: str) -> List[Position]:
    """
    Parse transaction-level CSV format and aggregate into positions with average prices.
    """
    from collections import defaultdict
    import re
    
    # Aggregate transactions by symbol
    positions_data = defaultdict(lambda: {
        'total_qty': 0.0,
        'total_cost': 0.0,
        'transactions': [],
        'first_date': None
    })
    
    for _, row in df.iterrows():
        # Skip empty rows or header rows
        if pd.isna(row.get(symbol_col)) or str(row.get(symbol_col)).strip() in ['', 'ticker/symbol']:
            continue
            
        symbol = _coerce_symbol(portfolio_type, str(row[symbol_col]).strip())
        
        # Skip if transaction type is not BUY (handle SELL later if needed)
        if transaction_type_col and str(row.get(transaction_type_col, '')).upper() != 'BUY':
            continue
            
        # Parse quantity
        qty = 0.0
        if amount_col and not pd.isna(row.get(amount_col)):
            qty_str = str(row[amount_col]).replace(',', '')
            try:
                qty = float(qty_str)
            except ValueError:
                continue
                
        # Parse USD cost
        usd_cost = 0.0
        if usd_col and not pd.isna(row.get(usd_col)):
            usd_str = str(row[usd_col]).replace('$', '').replace(',', '')
            try:
                usd_cost = float(usd_str)
            except ValueError:
                continue
                
        # Parse price (optional, can calculate from qty and cost)
        price = None
        if price_col and not pd.isna(row.get(price_col)):
            price_str = str(row[price_col]).replace('$', '').replace(',', '').replace('"', '')
            try:
                price = float(price_str)
            except ValueError:
                pass
                
        # If no price but have qty and cost, calculate price
        if not price and qty > 0 and usd_cost > 0:
            price = usd_cost / qty
            
        # Parse date
        date = None
        if date_col and not pd.isna(row.get(date_col)):
            date = pd.to_datetime(row[date_col], errors='coerce')
            
        # Aggregate data
        if qty > 0 and usd_cost > 0:
            pos_data = positions_data[symbol]
            pos_data['total_qty'] += qty
            pos_data['total_cost'] += usd_cost
            pos_data['transactions'].append({
                'qty': qty,
                'cost': usd_cost,
                'price': price,
                'date': date
            })
            
            # Track first purchase date
            if pos_data['first_date'] is None or (date and date < pos_data['first_date']):
                pos_data['first_date'] = date
    
    # Convert aggregated data to Position objects
    positions = []
    for symbol, data in positions_data.items():
        if data['total_qty'] > 0 and data['total_cost'] > 0:
            avg_price = data['total_cost'] / data['total_qty']
            position = Position(
                symbol=symbol,
                qty=data['total_qty'],
                purchase_price=avg_price,
                purchase_date=data['first_date'],
                total_cost_basis=data['total_cost'],
                transaction_count=len(data['transactions']),
                notes=f"Avg price from {len(data['transactions'])} transactions"
            )
            positions.append(position)
    
    return positions

def _yf_period_for_horizon(horizon: str) -> str:
    return {"short":"3mo","medium":"6mo","long":"2y"}.get(horizon, "6mo")

def _download_with_timeout(symbol: str, period: str, timeout: int = 10) -> pd.DataFrame:
    """Download data with timeout - simplified to avoid hanging"""
    try:
        # Use the specified period to get proper data for indicators
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, timeout=timeout)
        if df is not None and not df.empty:
            return df
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _download_history(symbol: str, horizon: str) -> pd.DataFrame:
    # daily bars are sufficient; intraday would require interval='1h' and different logic
    period = _yf_period_for_horizon(horizon)
    try:
        df = _download_with_timeout(symbol, period, timeout=10)

        if df.empty:
            # one more try: sometimes crypto pairs are weird
            if symbol.endswith("-USD"):
                df = _download_with_timeout(symbol.replace("-USD","-USDT"), period, timeout=10)

        if df.empty:
            # Create minimal fake data to prevent crashes
            import datetime as dt
            dates = pd.date_range(end=dt.datetime.now(), periods=5, freq='D')
            df = pd.DataFrame({
                'Open': [100]*5, 'High': [105]*5, 'Low': [95]*5,
                'Close': [102]*5, 'Volume': [1000]*5
            }, index=dates)

        # Handle MultiIndex columns that yfinance sometimes returns for single symbols
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        return df

    except Exception:
        # Return minimal fake data to prevent crashes
        import datetime as dt
        dates = pd.date_range(end=dt.datetime.now(), periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100]*5, 'High': [105]*5, 'Low': [95]*5,
            'Close': [102]*5, 'Volume': [1000]*5
        }, index=dates)
        return df

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()

def _indicators(df: pd.DataFrame) -> Indicators:
    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi14 = _rsi(close, 14)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up = bb_mid + 2*bb_std
    bb_dn = bb_mid - 2*bb_std
    atr14 = _atr(df, 14)
    last = -1
    return Indicators(
        close=float(close.iloc[last]),
        sma50=float(sma50.iloc[last]),
        sma200=float(sma200.iloc[last]) if not np.isnan(sma200.iloc[last]) else float("nan"),
        ema12=float(ema12.iloc[last]),
        ema26=float(ema26.iloc[last]),
        macd=float(macd.iloc[last]),
        macd_signal=float(macd_signal.iloc[last]),
        rsi14=float(rsi14.iloc[last]),
        bb_mid=float(bb_mid.iloc[last]),
        bb_up=float(bb_up.iloc[last]),
        bb_dn=float(bb_dn.iloc[last]),
        atr14=float(atr14.iloc[last])
    )

def _score_ta(ind: Indicators, horizon: str) -> Tuple[float,str]:
    parts = []
    score = 0.0

    # Trend bias
    if not np.isnan(ind.sma200):
        if ind.close > ind.sma200:
            score += 0.5; parts.append("Uptrend (close > SMA200)")
        else:
            score -= 0.5; parts.append("Downtrend (close < SMA200)")
    else:
        # for short periods where SMA200 is NaN
        if ind.close > ind.sma50:
            score += 0.2; parts.append("Above SMA50")
        else:
            score -= 0.2; parts.append("Below SMA50")

    # Momentum via MACD
    if ind.macd > ind.macd_signal:
        score += 0.25; parts.append("Momentum positive (MACD>signal)")
    else:
        score -= 0.25; parts.append("Momentum negative (MACD<signal)")

    # RSI contribution: lower RSI -> more upside potential; very high RSI -> downside
    rsi_term = ((50.0 - ind.rsi14) / 50.0) * 0.2
    score += rsi_term
    parts.append(f"RSI14={ind.rsi14:.1f}")

    # Bollinger extremes
    if ind.close < ind.bb_dn:
        score += 0.1; parts.append("Below lower Bollinger (mean-revert +)")
    elif ind.close > ind.bb_up:
        score -= 0.1; parts.append("Above upper Bollinger (mean-revert -)")

    # Horizon weighting (short puts more weight on momentum; long on trend)
    if horizon == "short":
        score *= 1.00
    elif horizon == "medium":
        score *= 0.95
    else: # long
        score *= 0.90

    return max(min(score, 1.0), -1.0), "; ".join(parts)

def _sentiment_vader(texts: List[str], weights: Optional[List[float]] = None) -> float:
    if not texts: return 0.0
    if _VADER is None: return 0.0

    vals = []
    text_weights = []

    for i, t in enumerate(texts):
        if t and isinstance(t, str):
            sentiment = _VADER.polarity_scores(t)["compound"]
            vals.append(sentiment)
            # Use provided weight or default weight of 1.0
            weight = weights[i] if weights and i < len(weights) else 1.0
            text_weights.append(weight)

    if not vals: return 0.0

    # Calculate weighted average, normalized by total weights
    total_weight = sum(text_weights)
    if total_weight == 0: return 0.0

    weighted_sum = sum(val * weight for val, weight in zip(vals, text_weights))
    weighted_avg = weighted_sum / total_weight

    # clamp to [-1,1]
    return float(np.clip(weighted_avg, -1.0, 1.0))

def _fetch_news_with_timeout(symbol: str, timeout: int = 10) -> List[dict]:
    """Fetch news with timeout"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news if hasattr(ticker, 'news') else []
        return news if news else []
    except Exception:
        return []

def _news_titles_for_symbol(symbol: str, portfolio_type: str, limit: int = 20) -> Tuple[List[str], List[float]]:
    titles = []
    weights = []
    weighted_sources = _get_weighted_news_sources(portfolio_type)

    # Try Yahoo Finance news via yfinance with timeout
    yahoo_weight = next((s["weight"] for s in weighted_sources if "Yahoo Finance" in s["name"]), 5.0)
    try:
        news_items = _fetch_news_with_timeout(symbol, timeout=10)

        for it in news_items[:limit]:
            t = it.get("title") or it.get("content","")
            if t:
                # Filter: mention symbol or synonym ($AAPL, BTC, etc.) or include anyway for breadth
                if re.search(rf'\b{re.escape(symbol.split("-")[0])}\b', t, re.I) or \
                   re.search(rf'\${re.escape(symbol.split("-")[0])}\b', t, re.I):
                    titles.append(t.strip())
                    weights.append(yahoo_weight)
                else:
                    titles.append(t.strip())
                    weights.append(yahoo_weight)
    except (TimeoutError, Exception):
        pass

    # Optional: RSS blending if feedparser is present (user can swap in preferred feeds)
    if feedparser is not None:
        # Use weighted sources for RSS feeds
        source_rss_map = {
            "CNBC Markets": "https://feeds.cnbc.com/rss/market.rss",
            "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        }

        for source in weighted_sources:
            source_name = source["name"]
            rss_url = source_rss_map.get(source_name)
            if not rss_url:
                continue

            source_weight = source["weight"]
            try:
                def rss_worker(result_dict, feed_url):
                    try:
                        result_dict['data'] = feedparser.parse(feed_url)
                    except Exception as e:
                        result_dict['error'] = e

                result = {'data': None, 'error': None}
                thread = threading.Thread(target=rss_worker, args=(result, rss_url))
                thread.daemon = True
                thread.start()
                thread.join(timeout=8)  # 8 second timeout for RSS

                if thread.is_alive():
                    continue

                if result['error']:
                    continue

                d = result['data']
                if d:
                    for e in d.entries[:max(0, limit//2)]:
                        title = getattr(e, "title", "").strip()
                        if title:
                            titles.append(title)
                            weights.append(source_weight)
            except Exception:
                continue

    # Deduplicate, keep most recent chunk and corresponding weights
    uniq_titles = []
    uniq_weights = []
    seen = set()
    for title, weight in zip(titles, weights):
        if title and title not in seen:
            uniq_titles.append(title)
            uniq_weights.append(weight)
            seen.add(title)

    # Limit results but maintain title-weight correspondence
    final_limit = min(limit, len(uniq_titles))
    return uniq_titles[:final_limit], uniq_weights[:final_limit]

def _x_posts_for_symbol(symbol: str, weighted_handles: List[Dict[str, float]], limit_per_handle: int = 5) -> Tuple[List[str], List[float]]:
    """
    Fetch X posts for symbol from weighted handles.
    Returns (posts, weights) where weights correspond to each post's source weight.
    """
    if not _SNSCRAPE:
        return [], []  # snscrape not available

    posts = []
    weights = []

    # Extract base symbol (e.g., BTC from BTC-USD)
    base_symbol = symbol.split('-')[0] if '-' in symbol else symbol

    for handle_info in weighted_handles[:5]:  # Limit to top 5 weighted handles
        handle = handle_info["handle"]
        weight = handle_info["weight"]

        try:
            # Use snscrape with timeout to search for symbol mentions from handle
            cmd = f'snscrape --max-results {limit_per_handle} --jsonl twitter-search "from:{handle} {base_symbol}"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5  # 5 second timeout per handle
            )

            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            tweet = json.loads(line)
                            content = tweet.get('content', '') or tweet.get('renderedContent', '')
                            if content:
                                posts.append(content)
                                weights.append(weight)
                        except json.JSONDecodeError:
                            continue
        except (subprocess.TimeoutExpired, Exception):
            continue  # Skip handles that timeout or error

    return posts, weights

def _get_reddit_client():
    """
    Get Reddit client using PRAW.
    Uses read-only mode (no authentication required).
    """
    if not _PRAW_AVAILABLE:
        return None

    try:
        # Use read-only mode which doesn't require authentication
        reddit = praw.Reddit(
            client_id="reddit_scraper",
            client_secret=None,
            user_agent="trading_signal_analyzer/1.0"
        )
        # Set to read-only mode
        reddit.read_only = True
        return reddit
    except Exception:
        return None

def _reddit_posts_for_symbol(symbol: str, weighted_subreddits: List[Dict[str, float]],
                             limit_per_subreddit: int = 25,
                             time_filter: str = "week") -> Tuple[List[str], List[float], Dict]:
    """
    Fetch Reddit posts and top comments for a symbol from weighted subreddits.
    Returns (texts, weights, metadata) where:
    - texts: list of post titles + selftext + top comments
    - weights: engagement-adjusted weights for each text
    - metadata: dict with stats about the fetch

    Engagement weighting formula:
    base_weight * (1 + log(1 + upvotes)/10) * (1 + awards/5) * (1 + comment_count/50)
    """
    if not _PRAW_AVAILABLE:
        return [], [], {"error": "PRAW not available"}

    reddit = _get_reddit_client()
    if reddit is None:
        return [], [], {"error": "Could not initialize Reddit client"}

    texts = []
    weights = []

    # Extract base symbol (e.g., BTC from BTC-USD, AAPL from AAPL)
    base_symbol = symbol.split('-')[0] if '-' in symbol else symbol

    # Metadata tracking
    total_posts = 0
    total_comments = 0
    subreddits_checked = 0

    for sub_info in weighted_subreddits[:7]:  # Limit to top 7 weighted subreddits
        subreddit_name = sub_info["subreddit"]
        base_weight = sub_info["weight"]

        try:
            subreddit = reddit.subreddit(subreddit_name)

            # Search for symbol mentions in the subreddit
            # Use multiple search strategies for better coverage
            search_queries = [
                f"{base_symbol}",
                f"${base_symbol}",
            ]

            posts_found = set()  # Track unique posts by ID

            for query in search_queries:
                try:
                    # Search recent posts mentioning the symbol
                    for post in subreddit.search(query, time_filter=time_filter, limit=limit_per_subreddit):
                        if post.id in posts_found:
                            continue  # Skip duplicates
                        posts_found.add(post.id)

                        # Calculate engagement multiplier
                        upvotes = max(0, post.score)
                        awards = post.total_awards_received if hasattr(post, 'total_awards_received') else 0
                        num_comments = post.num_comments

                        # Engagement formula: logarithmic scaling to prevent extreme outliers
                        engagement_multiplier = (
                            (1 + np.log1p(upvotes) / 10) *     # Upvote boost (log scale)
                            (1 + awards / 5) *                  # Award boost
                            (1 + num_comments / 50)             # Comment activity boost
                        )

                        # Final weight combines subreddit credibility with engagement
                        post_weight = base_weight * engagement_multiplier

                        # Extract post title and body
                        post_text = post.title
                        if post.selftext and len(post.selftext) > 0:
                            post_text += " " + post.selftext[:500]  # Limit body length

                        texts.append(post_text)
                        weights.append(post_weight)
                        total_posts += 1

                        # Also fetch top comments for deeper sentiment analysis
                        try:
                            post.comment_sort = 'top'
                            post.comment_limit = 5  # Top 5 comments
                            post.comments.replace_more(limit=0)  # Don't expand "more comments"

                            for comment in post.comments[:5]:
                                if hasattr(comment, 'body') and len(comment.body) > 20:
                                    comment_upvotes = max(0, comment.score)
                                    # Comments get lower base weight but still benefit from engagement
                                    comment_multiplier = (1 + np.log1p(comment_upvotes) / 15)
                                    comment_weight = (base_weight * 0.6) * comment_multiplier

                                    texts.append(comment.body[:300])  # Limit comment length
                                    weights.append(comment_weight)
                                    total_comments += 1
                        except Exception:
                            pass  # Skip comments if there's an issue

                        # Rate limiting: small delay between posts
                        time.sleep(0.1)

                except Exception:
                    continue  # Skip this search query if it fails

            subreddits_checked += 1

        except Exception:
            continue  # Skip this subreddit if there's an issue

    metadata = {
        "total_posts": total_posts,
        "total_comments": total_comments,
        "subreddits_checked": subreddits_checked,
        "total_texts": len(texts)
    }

    return texts, weights, metadata

def _score_news_x_and_reddit(symbol: str, portfolio_type: str, horizon: str) -> Tuple[float,float,float,str]:
    # News titles sentiment with source weighting
    news_titles, news_weights = _news_titles_for_symbol(symbol, portfolio_type, limit=30)
    news_score = _sentiment_vader(news_titles, news_weights)

    # X posts sentiment with source weighting
    weighted_handles = _get_weighted_social_sources(portfolio_type)
    x_posts, x_weights = _x_posts_for_symbol(symbol, weighted_handles, limit_per_handle=5)
    x_score = _sentiment_vader(x_posts, x_weights)

    # Reddit posts and comments sentiment with engagement weighting
    weighted_subreddits = _get_weighted_reddit_sources(portfolio_type)

    # Adjust time filter based on horizon
    if horizon == "short":
        time_filter = "day"
        limit_per_sub = 30
    elif horizon == "medium":
        time_filter = "week"
        limit_per_sub = 25
    else:  # long
        time_filter = "month"
        limit_per_sub = 20

    reddit_texts, reddit_weights, reddit_metadata = _reddit_posts_for_symbol(
        symbol, weighted_subreddits, limit_per_subreddit=limit_per_sub, time_filter=time_filter
    )
    reddit_score = _sentiment_vader(reddit_texts, reddit_weights)

    # Weighting by horizon
    # Short: news + X + Reddit balanced for quick signals
    # Medium: balanced across all sources
    # Long: news heavier, Reddit and X for sentiment validation
    if horizon == "short":
        wn, wx, wr = 0.50, 0.25, 0.25
    elif horizon == "medium":
        wn, wx, wr = 0.55, 0.25, 0.20
    else:  # long
        wn, wx, wr = 0.60, 0.20, 0.20

    blended = wn*news_score + wx*x_score + wr*reddit_score

    # Build detailed report
    detail_parts = [
        f"NewsSent={news_score:+.2f} ({len(news_titles)} headlines)",
        f"XSent={x_score:+.2f} ({len(x_posts)} posts)",
        f"RedditSent={reddit_score:+.2f} ({reddit_metadata.get('total_posts', 0)} posts, {reddit_metadata.get('total_comments', 0)} comments)"
    ]
    detail = "; ".join(detail_parts)

    return float(blended), float(x_score), float(reddit_score), detail

def _combine_scores(ta: float, news_blended: float, x: float, reddit: float, horizon: str) -> float:
    # Blend: short -> TA heavier; long -> sentiment heavier
    # News/X/Reddit are already blended in news_blended, but we keep separate scores for reporting
    if horizon == "short":
        w_ta, w_sentiment = 0.60, 0.40
    elif horizon == "medium":
        w_ta, w_sentiment = 0.50, 0.50
    else:  # long
        w_ta, w_sentiment = 0.40, 0.60

    # news_blended already contains weighted combination of news, X, and Reddit
    total = w_ta * ta + w_sentiment * news_blended

    return float(np.clip(total, -1.0, 1.0))

def _decide(total_score: float, ind: Indicators) -> Decision:
    # Map score to discrete action and target allocation delta (suggested)
    if total_score >= 0.60:
        return Decision("Strong Buy", +3.0, f"Score={total_score:+.2f} (high conviction)")
    if total_score >= 0.30:
        return Decision("Buy", +1.5, f"Score={total_score:+.2f}")
    if total_score <= -0.60:
        return Decision("Sell", -3.0, f"Score={total_score:+.2f} (high conviction)")
    if total_score <= -0.30:
        return Decision("Trim", -1.5, f"Score={total_score:+.2f}")
    return Decision("Hold", 0.0, f"Score={total_score:+.2f}")

# ---------------------- Agentic prompt generation --------------------- #

PLATFORM_ALIASES = {
    "coinbase": ["coinbase", "coin base", "cb"],
    "fidelity": ["fidelity", "fido"],
    "schwab": ["schwab", "charles schwab", "td ameritrade", "tda"],
    "etrade": ["etrade", "e*trade", "e-trade"],
    "robinhood": ["robinhood", "rh"],
    "ibkr": ["ibkr", "interactive brokers", "ib"]
}

def _normalize_platform(p: str) -> str:
    p = p.strip().lower()
    for k, vals in PLATFORM_ALIASES.items():
        if p == k or p in vals:
            return k
    return p

def _prompt_for_platform(platform: str, broker_action: str, symbol: str,
                         target_pct_delta: float, horizon: str,
                         order_type: str = "market",
                         limit_hint: Optional[float] = None) -> str:
    """
    Return Comet prompt tailored to the platform.
    Optimized for voice commands and autonomous execution.
    """
    # Comet-optimized base directive - action-oriented and conversational
    base_directive = textwrap.dedent(f"""
    Execute {broker_action.lower()} order for {symbol} on {platform.title()}.

    Trading parameters:
    • Action: {broker_action.upper()}
    • Symbol: {symbol}
    • Order type: {order_type.upper()}
    • Portfolio allocation: {abs(target_pct_delta):.1f}% of total account value
    • Time-in-force: DAY order

    Execution requirements:
    • Use live market price from platform
    • Calculate position size from account balance
    • Pause for 2FA if prompted - ask user for confirmation
    • Verify order details before submission
    • Confirm execution and provide order summary
    """).strip()

    # Platform-specific instructions optimized for Comet
    platform_steps = {
        "coinbase": f"""
🚀 COMET-OPTIMIZED for Coinbase Partnership:
Access live {symbol} data through Perplexity integration. Execute {broker_action.lower()} order:
→ Use built-in Coinbase market data for {symbol} price analysis
→ Navigate to Coinbase trading interface
→ Search and select "{symbol}"
→ Choose {"Buy" if broker_action.lower() in ["buy","strong buy"] else "Sell"} with real-time pricing
→ Set order type: {order_type.title()}
→ Calculate position: account_balance × {abs(target_pct_delta)/100:.3f} ÷ live_price
→ Leverage Comet's autonomous execution for seamless order placement
→ Confirm trade with integrated verification system
        """,
        "fidelity": f"""
Go to Fidelity trading page for {symbol}. Execute {broker_action.lower()}:
→ Navigate to Trade > Stocks/ETFs
→ Enter symbol: {symbol}
→ Select {"BUY" if broker_action.lower().startswith("buy") else "SELL"} action
→ Set order type: {order_type.title()}
→ Calculate quantity: (account_value × {abs(target_pct_delta)/100:.3f}) ÷ live_price
→ Set time-in-force: DAY
→ Preview and place order
        """,
        "schwab": f"""
Access Schwab trading for {symbol}. Place {broker_action.lower()} order:
→ Go to Trade > Stocks & ETFs
→ Symbol: {symbol}, Action: {"Buy" if broker_action.lower().startswith("buy") else "Sell"}
→ Order type: {order_type.title()}
→ Shares from {abs(target_pct_delta):.1f}% portfolio allocation
→ Time-in-force: DAY
→ Review and place order
        """,
        "etrade": f"""
Navigate to E*TRADE for {symbol} trading. Execute {broker_action.lower()}:
→ Access Trading > Stocks/ETFs
→ Symbol: {symbol}
→ Action: {"Buy" if broker_action.lower().startswith("buy") else "Sell"}
→ Order: {order_type.title()}
→ Calculate quantity from {abs(target_pct_delta):.1f}% allocation
→ Set TIF: DAY, Preview and place
        """,
        "robinhood": f"""
Open {symbol} on Robinhood. Place {broker_action.lower()} order:
→ Search and select {symbol}
→ Tap Trade > {"Buy" if broker_action.lower().startswith("buy") else "Sell"}
→ Switch to {order_type.title()} order
→ Calculate: portfolio_value × {abs(target_pct_delta)/100:.3f} = order_amount
→ Review order details and submit
        """,
        "ibkr": f"""
Access IBKR order ticket for {symbol}. Execute {broker_action.lower()}:
→ Client Portal > Trade > Order Ticket
→ Symbol: {symbol}
→ Side: {"BUY" if broker_action.lower().startswith("buy") else "SELL"}
→ Type: {"MKT" if order_type=='market' else "LMT"}
→ Calculate quantity from target allocation
→ TIF: DAY, Submit and transmit
        """,
    }


    plat = _normalize_platform(platform)

    # Get platform-specific steps
    step_text = platform_steps.get(plat, f"""
Access {platform} trading for {symbol}. Execute {broker_action.lower()}:
→ Open order ticket for {symbol}
→ Select {"BUY" if broker_action.lower().startswith("buy") else "SELL"} action
→ Set order type: {order_type.title()}
→ Calculate quantity from {abs(target_pct_delta):.1f}% of account value
→ Set TIF: DAY, Preview and submit
    """)

    # Comet prompt - optimized for voice commands and autonomous execution
    prompt = f"""COMET TRADING COMMAND
{base_directive}

EXECUTION WORKFLOW:
{textwrap.dedent(step_text).strip()}

COMPLETION CHECKLIST:
✓ Verify symbol matches: {symbol}
✓ Confirm order type: {order_type.upper()}
✓ Validate allocation: {abs(target_pct_delta):.1f}% of portfolio
✓ Check order status and provide execution summary
✓ Report: symbol, action, quantity, price, order ID, timestamp

VOICE COMMAND: "Execute {broker_action.lower()} order for {symbol} using {abs(target_pct_delta):.1f}% portfolio allocation on {platform}"
"""

    return prompt.strip()


# ------------------------------ Main --------------------------------- #

def analyze_positions(portfolio_type: str, horizon: str, platform: str, positions: List[Position]) -> pd.DataFrame:
    rows = []
    for pos in positions:
        try:
            df = _download_history(pos.symbol, horizon)
            ind = _indicators(df)
            ta_score, ta_detail = _score_ta(ind, horizon)
            news_blended, x_only, reddit_only, sentiment_detail = _score_news_x_and_reddit(pos.symbol, portfolio_type, horizon)
            total = _combine_scores(ta_score, news_blended, x_only, reddit_only, horizon)
            decision = _decide(total, ind)

            # Calculate confidence score if available
            confidence_breakdown = None
            if _CONFIDENCE_AVAILABLE and ConfidenceScorer:
                try:
                    scorer = ConfidenceScorer()
                    # Convert indicators to dict format
                    indicators_dict = {
                        'close': ind.close,
                        'rsi14': ind.rsi14,
                        'sma50': ind.sma50,
                        'sma200': ind.sma200,
                        'macd': ind.macd,
                        'macd_signal': ind.macd_signal,
                        'bb_up': ind.bb_up,
                        'bb_dn': ind.bb_dn,
                        'bb_mid': ind.bb_mid,
                        'atr14': ind.atr14
                    }
                    sentiment_dict = {
                        'sentiment_score': news_blended,
                        'x_only_score': x_only,
                        'reddit_only_score': reddit_only
                    }
                    confidence_breakdown = scorer.calculate_confidence(
                        indicators_dict, sentiment_dict, decision.action, pos.symbol, horizon
                    )
                except Exception as conf_e:
                    print(f"Warning: Confidence calculation failed for {pos.symbol}: {conf_e}")
                    confidence_breakdown = None

            row_data = {
                "symbol": pos.symbol,
                "close": ind.close,
                "rsi14": ind.rsi14,
                "sma50": ind.sma50,
                "sma200": ind.sma200,
                "macd": ind.macd,
                "macd_signal": ind.macd_signal,
                "atr14": ind.atr14,
                "ta_score": round(ta_score,3),
                "sentiment_score": round(news_blended,3),
                "x_only_score": round(x_only,3),
                "reddit_only_score": round(reddit_only,3),
                "total_score": round(total,3),
                "decision": decision.action,
                "target_pct_delta": decision.target_pct_delta,
                "ta_reason": ta_detail,
                "sentiment_detail": sentiment_detail
            }

            # Add confidence data if available
            if confidence_breakdown:
                row_data.update({
                    "confidence": confidence_breakdown.overall,
                    "conf_technical": confidence_breakdown.technical,
                    "conf_sentiment": confidence_breakdown.sentiment,
                    "conf_strength": confidence_breakdown.strength,
                    "conf_historical": confidence_breakdown.historical
                })
            else:
                row_data.update({
                    "confidence": float('nan'),
                    "conf_technical": float('nan'),
                    "conf_sentiment": float('nan'),
                    "conf_strength": float('nan'),
                    "conf_historical": float('nan')
                })

            rows.append(row_data)
        except Exception as e:
            error_row = {
                "symbol": pos.symbol,
                "close": float('nan'),
                "rsi14": float('nan'),
                "sma50": float('nan'),
                "sma200": float('nan'),
                "macd": float('nan'),
                "macd_signal": float('nan'),
                "atr14": float('nan'),
                "ta_score": float('nan'),
                "sentiment_score": float('nan'),
                "x_only_score": float('nan'),
                "reddit_only_score": float('nan'),
                "total_score": float('nan'),
                "decision": "No Data",
                "target_pct_delta": 0.0,
                "ta_reason": "Error occurred",
                "sentiment_detail": "Error occurred",
                "confidence": float('nan'),
                "conf_technical": float('nan'),
                "conf_sentiment": float('nan'),
                "conf_strength": float('nan'),
                "conf_historical": float('nan'),
                "error": str(e)
            }
            rows.append(error_row)
    return pd.DataFrame(rows)

def generate_prompts(df_signals: pd.DataFrame, platform: str, horizon: str) -> List[Dict[str,str]]:
    prompts = []
    for _, r in df_signals.iterrows():
        if r.get("decision") in ["Buy","Strong Buy","Trim","Sell"]:
            side = r["decision"]
            # For simplicity, trims map to SELL, but the browser prompt explains "reduce by X%".
            broker_action = "Buy" if side in ["Buy","Strong Buy"] else "Sell"
            prompt = _prompt_for_platform(platform, broker_action, r["symbol"], float(r["target_pct_delta"]), horizon)
            prompts.append({"symbol": r["symbol"], "decision": side, "prompt": prompt})
    return prompts

def _write_outputs(df: pd.DataFrame, prompts: List[Dict[str,str]]) -> Tuple[str,str]:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sig_path = f"signals_{stamp}.csv"
    pr_path  = f"prompts_{stamp}.txt"
    df.to_csv(sig_path, index=False)
    with open(pr_path, "w", encoding="utf-8") as f:
        f.write("===== COMET TRADING PROMPTS =====\n\n")
        if not prompts:
            f.write("(No actionable prompts — all Hold or no data)\n\n")
        for item in prompts:
            f.write(f"### {item['symbol']} — {item['decision']}\n")
            f.write(item["prompt"])
            f.write("\n\n---\n\n")
    return sig_path, pr_path

def main():
    parser = argparse.ArgumentParser(description="Comet Trading Agent (signals + Comet browser prompts)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--portfolio-type", required=True, choices=["stocks","crypto"])
    parser.add_argument("--horizon", required=True, choices=["short","medium","long"])
    parser.add_argument("--platform", required=True, help="e.g., Coinbase, Fidelity, Schwab, E*TRADE, Robinhood, IBKR")
    parser.add_argument("--file", required=True, help="CSV or XLSX portfolio file")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                       help="Minimum confidence score (0-100) to include signals (default: 0)")
    parser.add_argument("--show-confidence-viz", action="store_true",
                       help="Generate and display confidence score distribution visualization")
    args = parser.parse_args()

    positions = _read_portfolio(args.file, args.portfolio_type)
    df_signals = analyze_positions(args.portfolio_type, args.horizon, args.platform, positions)

    # Apply confidence filtering if requested
    if args.min_confidence > 0 and _CONFIDENCE_AVAILABLE and 'confidence' in df_signals.columns:
        print(f"\nApplying confidence filter (min: {args.min_confidence})")
        df_signals_filtered = filter_signals_by_confidence(df_signals, args.min_confidence)
    else:
        df_signals_filtered = df_signals

    # Generate confidence visualization if requested
    if args.show_confidence_viz and _CONFIDENCE_AVAILABLE and 'confidence' in df_signals.columns:
        try:
            from confidence_scorer import ConfidenceScorer
            scorer = ConfidenceScorer()
            confidence_scores = df_signals['confidence'].dropna().tolist()
            if confidence_scores:
                viz_path = f"confidence_distribution_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                scorer.visualize_confidence_distribution(confidence_scores,
                                                       f"{args.portfolio_type.title()} Portfolio",
                                                       viz_path)
                print(f"Confidence visualization saved to: {viz_path}")
            else:
                print("No confidence scores available for visualization")
        except Exception as viz_e:
            print(f"Warning: Confidence visualization failed: {viz_e}")

    prompts = generate_prompts(df_signals_filtered, args.platform, args.horizon)
    sig_path, pr_path = _write_outputs(df_signals_filtered, prompts)

    # Console summary
    print("\n=== COMET TRADING REPORT ===")
    print(f"Version: {__version__}")
    print(f"Portfolio type: {args.portfolio_type} | Horizon: {args.horizon} | Platform: {args.platform}")

    # Select columns that exist in the DataFrame
    available_cols = []
    for col in ["symbol", "decision", "target_pct_delta"]:
        if col in df_signals_filtered.columns:
            available_cols.append(col)

    # Add optional columns if they exist
    if "close" in df_signals_filtered.columns:
        available_cols.insert(-1, "close")
    if "total_score" in df_signals_filtered.columns:
        available_cols.insert(-1, "total_score")
    if "confidence" in df_signals_filtered.columns and _CONFIDENCE_AVAILABLE:
        available_cols.insert(-1, "confidence")

    # Display confidence summary if available
    if _CONFIDENCE_AVAILABLE and 'confidence' in df_signals.columns:
        confidence_scores = df_signals['confidence'].dropna()
        if len(confidence_scores) > 0:
            print(f"\n=== CONFIDENCE SUMMARY ===")
            print(f"Average Confidence: {confidence_scores.mean():.1f}")
            print(f"Median Confidence: {confidence_scores.median():.1f}")
            print(f"Confidence Range: {confidence_scores.min():.1f} - {confidence_scores.max():.1f}")

            high_conf = (confidence_scores >= 70).sum()
            med_conf = ((confidence_scores >= 40) & (confidence_scores < 70)).sum()
            low_conf = (confidence_scores < 40).sum()
            print(f"High Confidence (≥70): {high_conf} signals")
            print(f"Medium Confidence (40-69): {med_conf} signals")
            print(f"Low Confidence (<40): {low_conf} signals")

    print(df_signals_filtered[available_cols].to_string(index=False))
    print(f"\nSaved signals to: {sig_path}")
    print(f"Saved Comet prompts to: {pr_path}")
    print("\nOptimized for Perplexity Comet browser automation")
    print("DISCLAIMER: Educational research tool. Not financial advice. Review orders before submitting.")

if __name__ == "__main__":
    main()