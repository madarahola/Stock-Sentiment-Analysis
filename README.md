# Stock-Sentiment-Analysis
 Automate investment insights by combining real-time stock news, financial metrics, and sentiment analysis to rate a stock on a scale of 1–10.


# ----- Code Analysis ---------

🔶 1. Load FinBERT model 

FinBERT is a financial sentiment analysis model fine-tuned on finance-related news/articles.

tokenizer prepares text for input to the model.

model is the actual neural network that predicts positive, neutral, or negative sentiment.

🔶 2. Fetch Financial News Articles 

Uses the newspaper3k library to scrape news articles from Google/Moneycontrol based on the company name.

It tries to download and parse up to limit articles.

Returns the raw text of each article.

🔶 3. Analyze News Sentiment Using FinBERT

Converts news text into tokens that the model understands.

Predicts the probability of negative / neutral / positive.

Returns the label and probabilities, e.g.,: ('positive', [0.12, 0.18, 0.70])

🔶 4. Get Stock Financial Metrics from Yahoo Finance 

Uses yfinance to get:

Current stock price

P/E ratio

EPS (earnings per share)

Analyst recommendation

This helps assess the financial fundamentals of the company.

🔶 5. Generate Investment Score

Combines:

Sentiment positivity

Financial metrics (low P/E is good, high EPS is good)

Applies weighted logic to compute a score out of 10.

Returns the final rating for the stock.

🔶 6. Master Agent Function: analyze_stock()

Here's what this part does:
Fetches news → fetch_articles

Analyzes each article’s sentiment → analyze_sentiment

Averages the sentiment scores

Fetches financials → get_stock_financials

Combines both into a final score → generate_score

Formats the output cleanly with headings:


# 🧾 Output Format (clean & readable):

1. **Sentiment Analysis**:
   - Average Sentiment: Positive
   - Positive: 0.70, Neutral: 0.18, Negative: 0.12
   - Articles Analyzed: 5

2. **Financial Data**:
   - Current Price: ₹1287.45
   - P/E Ratio: 28.56
   - EPS: 14.2
   - Analyst Recommendation: buy

3. **Consolidated Analysis**:
   - Final Score (1–10): 8.2
   - Verdict: 👍 Buy


