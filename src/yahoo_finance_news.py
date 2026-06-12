import yfinance as yf

# Enter the ticker symbol (e.g., AAPL)
ticker_symbol = "NVDA"
ticker = yf.Ticker(ticker_symbol)

# Fetch the most recent news articles
news_items = ticker.news

# Print out titles and publication dates
for article in news_items:
    title = article.get("content", {}).get("title")
    pub_date = article.get("content", {}).get("pubDate")
    summary = article.get("content", {}).get("summary")
    print(f"Title: {title}")
    print(f"Published: {pub_date}")
    print(f"Summary: {summary}\n")

print(f"Article count: {len(news_items)}")
