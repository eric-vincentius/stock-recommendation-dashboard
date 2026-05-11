from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()

    labels = ["negative", "neutral", "positive"]

    return labels[predicted_class], probs[0][predicted_class].item()

# =========================
# 1. CONFIG
# =========================
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://www.cnbcindonesia.com/market/indeks/5?page="
MAX_PAGES = 40

# =========================
# 2. STOCK DICTIONARY
# =========================
stock_dict = {
    "MAPI": ["mapi", "mitra adi perkasa"],
    "ACES": ["aces", "ace hardware"],
    "ADRO": ["adro", "adaro energy"],
    "AKRA": ["akra", "akr corporindo"],
    "AMRT": ["amrt", "alfamart", "sumber alfaria"],
    "ASII": ["asii", "astra international", "astra"],
    "BBNI": ["bbni", "bni", "bank negara indonesia"],
    "CPIN": ["cpin", "charoen pokphand"],
    "EXCL": ["excl", "xl axiata"],
    "GGRM": ["ggrm", "gudang garam"],
    "ICBP": ["icbp", "indofood cbp"],
    "INCO": ["inco", "vale indonesia"],
    "INDF": ["indf", "indofood"],
    "INKP": ["inkp", "indah kiat"],
    "INTP": ["intp", "indocement"],
    "ITMG": ["itmg", "indo tambangraya"],
    "KLBF": ["klbf", "kalbe farma"],
    "MEDC": ["medc", "medco energi"],
    "PGAS": ["pgas", "perusahaan gas negara", "pgn"],
    "PTBA": ["ptba", "bukit asam"],
    "SMGR": ["smgr", "semen indonesia"],
    "UNTR": ["untr", "united tractors"],
    "UNVR": ["unvr", "unilever indonesia", "unilever"],
    "ANTM": ["antm", "antam"],
    "BBCA": ["bbca", "bca", "bank central asia"],
    "BBRI": ["bbri", "bri", "bank rakyat indonesia"],
    "BMRI": ["bmri", "mandiri", "bank mandiri"],
    "BRPT": ["brpt", "barito pacific"],
    "TLKM": ["tlkm", "telkom", "telkom indonesia"]
}

# =========================
# 3. GET ARTICLES (TITLE ONLY)
# =========================
def get_market_articles(page_url):
    res = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    articles_data = []

    articles = soup.find_all("article")

    for art in articles:
        a_tag = art.find("a", href=True)
        title_tag = art.find("h2")

        if a_tag and title_tag:
            url = a_tag["href"]
            title = title_tag.text.strip()

            if url.startswith("https://www.cnbcindonesia.com/market/"):
                articles_data.append({
                    "title": title,
                    "url": url
                })

    return articles_data

# =========================
# 4. MATCH STOCK
# =========================
def match_stock(text):
    text = text.lower()
    matched = []

    for ticker, keywords in stock_dict.items():
        for k in keywords:
            if f" {k} " in f" {text} ":
                matched.append(ticker)
                break

    return matched

# =========================
# 5. SIMPLE SCORING (TITLE ONLY)
# =========================
def compute_score(title, keywords):
    title = title.lower()
    score = 0

    if any(k in title for k in keywords):
        score += 2

    return score

def sentiment_to_weighted_score(label, confidence):
    base = {
        "negative": -1,
        "neutral": 0,
        "positive": 1
    }.get(label, 0)

    return base * confidence

# =========================
# 6. MAIN PIPELINE
# =========================
def run_pipeline():
    all_articles = []
    seen_titles = set()

    for page in range(1, MAX_PAGES + 1):
        page_url = f"{BASE_URL}{page}"
        print(f"Scraping page: {page_url}")

        articles = get_market_articles(page_url)
        print(f"Found {len(articles)} articles")

        for art in articles:
            if art["title"] in seen_titles:
                continue

            seen_titles.add(art["title"])

            matched_stocks = match_stock(art["title"])

            for ticker in matched_stocks:
                keywords = stock_dict[ticker]
                score = compute_score(art["title"], keywords)
                sentiment_label, sentiment_score = predict_sentiment(art["title"])

                df["sentiment_weighted"] = df.apply(
                    lambda x: sentiment_to_weighted_score(x["sentiment"], x["confidence"]),
                    axis=1
                )

                if score >= 2:
                    all_articles.append({
                        "ticker": ticker,
                        "title": art["title"],
                        "sentiment": sentiment_label,
                        "score": score,
                        "url": art["url"]
                    })

        time.sleep(1)  # avoid blocking

    df = pd.DataFrame(all_articles)
    df.drop_duplicates(inplace=True)

    return df

# =========================
# 7. RUN
# =========================
if __name__ == "__main__":
    df = run_pipeline()

    print(df.head())
    print(f"Total results: {len(df)}")

    df.to_csv("data/news.csv", index=False)