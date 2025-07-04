# Loading FinBERT Model 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# ----------------------------------------- Browsing google and Money Control --------------------------------------------

from newspaper import Article
import newspaper

def fetch_articles(query, limit=5):
    paper = newspaper.build(f'https://www.google.com/search?q={query}+site:moneycontrol.com', memoize_articles=False)
    articles = []
    for content in paper.articles[:limit]:
        try:
            content.download()
            content.parse()
            articles.append(content.text)
        except:
            continue
    return articles


# ----------------------------------------- Sentiment Analyzer --------------------------------------------------------------

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    labels = ['negative', 'neutral', 'positive']
    return labels[probs.argmax()], probs[0].tolist()


# ---------------- Get Stock Financials ----------------------------------------------

import yfinance as yf

def get_stock_financials(ticker):
    stock = yf.Ticker(ticker)
    data = {
        "price": stock.history(period="1mo")["Close"].iloc[-1],
        "pe_ratio": stock.info.get("trailingPE"),
        "eps": stock.info.get("trailingEps"),
        "recommendation": stock.info.get("recommendationKey")
    }
    return data


# --------------------------Scoring Logic ---------------------------------------------


def generate_score(sentiment_probs, financials):
    sentiment_weight = sentiment_probs[2] * 10  # Positive probability
    pe_factor = 1 if financials["pe_ratio"] and financials["pe_ratio"] < 30 else 0.5
    eps_factor = 1 if financials["eps"] and financials["eps"] > 0 else 0.3

    score = (sentiment_weight * 0.6) + (pe_factor * 3) + (eps_factor * 2)
    return min(round(score, 2), 10)



# ---------------------- Final Agent ---------------------------------------------------

def analyze_stock(ticker, company_name):
    print(f"🔍 Analyzing {company_name} ({ticker})...\n")
    
    # 1. News Sentiment
    articles = fetch_articles(company_name)
    sentiments = [analyze_sentiment(article) for article in articles]
    avg_sentiment = np.mean([s[1] for s in sentiments], axis=0)
    sentiment_label = ['negative', 'neutral', 'positive'][np.argmax(avg_sentiment)]

    # 2. Financials
    financials = get_stock_financials(ticker)
    score = generate_score(avg_sentiment, financials)

    # 3. Format clean output
    sentiment_section = f"""\
1. **Sentiment Analysis**:
   - Average Sentiment: **{sentiment_label.capitalize()}**
   - Positive: {avg_sentiment[2]:.2f}, Neutral: {avg_sentiment[1]:.2f}, Negative: {avg_sentiment[0]:.2f}
   - Articles Analyzed: {len(articles)}
"""

    financial_section = f"""\
2. **Financial Data**:
   - Current Price: ₹{financials.get('price', 'N/A'):.2f}
   - P/E Ratio: {financials.get('pe_ratio', 'N/A')}
   - EPS: {financials.get('eps', 'N/A')}
   - Analyst Recommendation: {financials.get('recommendation', 'N/A')}
"""

    consolidated_section = f"""\
3. **Consolidated Analysis**:
   - Final Score (1-10): **{score}**
   - Verdict: {"👍 Buy" if score >= 7 else "⚠️ Hold" if score >= 5 else "❌ Avoid"}
"""

    return sentiment_section + financial_section + consolidated_section


result = analyze_stock("DEEPAKFERT.NS", "Deepak Fertilisers And Petrochemicals Corporation Limited")
print(result)



