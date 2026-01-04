# Reddit Subreddit Source Weighting Configuration

This file configures which subreddits to analyze and their credibility weights for trading signals.

## How It Works

- Each subreddit gets a weight from 1-10 (10 = highest credibility)
- Weights are multiplied by engagement metrics (upvotes, awards, comments)
- Higher weighted subreddits have more influence on trading signals
- The system analyzes both posts and top comments for deeper sentiment

## Format

```
subreddit, weight
```

## Stock Market Subreddits

### High Quality Analysis (8-10)
- For fundamental analysis, professional discussion, institutional-quality insights

```
subreddit, weight
SecurityAnalysis, 9
ValueInvesting, 8
investing, 8
Dividends, 7
```

### Active Trading Communities (5-7)
- For market sentiment, active trading discussion, momentum signals

```
stocks, 7
StockMarket, 6
options, 6
Daytrading, 5
```

### Retail Sentiment (3-5)
- For contrarian signals, retail enthusiasm, meme stock tracking

```
wallstreetbets, 4
```

## Cryptocurrency Subreddits

### High Quality Research (8-10)
- For protocol analysis, technical research, institutional-level discussion

```
Bitcoin, 8
ethereum, 8
CryptoMarkets, 7
defi, 7
```

### Active Communities (5-7)
- For market sentiment, trading signals, project discussion

```
CryptoCurrency, 6
BitcoinMarkets, 6
altcoin, 5
binance, 5
```

### Emerging Trends (3-5)
- For trend detection, early signals, speculative plays

```
NFT, 4
```

## Customization Tips

1. **Higher weights (8-10)**: Use for subreddits with:
   - Professional moderation
   - Quality post requirements
   - Fundamental analysis focus
   - Long-term investment perspective

2. **Medium weights (5-7)**: Use for:
   - Active trading communities
   - Balanced discussion
   - Diverse perspectives
   - Moderate noise levels

3. **Lower weights (3-4)**: Use for:
   - High volume/noise ratio
   - Speculative discussion
   - Contrarian indicators
   - Meme-heavy content

4. **Add custom subreddits**:
   - Format: `subreddit_name, weight`
   - No r/ prefix needed
   - Weight must be between 1-10

## Engagement Weighting Formula

Each post's weight is multiplied by engagement:
```
final_weight = base_weight × (1 + log(upvotes)/10) × (1 + awards/5) × (1 + comments/50)
```

This ensures:
- Highly upvoted posts carry more weight
- Awarded posts (premium engagement) boost signal
- Discussion depth (comments) indicates importance
- Logarithmic scaling prevents outlier dominance

## Notes

- The system automatically fetches posts and top comments
- Time horizons adjust search filters (day/week/month)
- Comment sentiment gets 60% of post weight
- Duplicate posts are automatically filtered
- Rate limiting prevents API throttling
