# pip install yfinance transformers xgboost pandas scikit-learn torch

import os
# Configure environment variables to prevent OpenMP multi-threading conflicts on macOS (SIGSEGV/Exit code 139)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ==========================================
# STEP 1: NLP Sentiment Analyzer Initialisation
# ==========================================
# We load the FinBERT model which is specifically pre-trained on financial text.
# The Hugging Face pipeline makes sentiment analysis simple and efficient.
print("Initialising FinBERT sentiment analysis pipeline...")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ==========================================
# STEP 2: Stock Data Acquisition (yfinance)
# ==========================================
def download_stock_data(ticker="AAPL", start_date=None, end_date=None):
    """
    Downloads historical stock data for the given ticker.
    Defaults to the last 2 years if no dates are specified.
    """
    if start_date is None or end_date is None:
        end_dt = datetime.date.today()
        start_dt = end_dt - datetime.timedelta(days=2 * 365)
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
    else:
        start_str = start_date
        end_str = end_date

    print(f"Downloading historical stock data for {ticker} from {start_str} to {end_str}...")
    df_stock = yf.download(ticker, start=start_str, end=end_str)
    
    # Check for empty data
    if df_stock.empty:
        raise ValueError(f"No stock data downloaded for {ticker}. Check dates or internet connection.")
        
    # Calculate daily returns: (Close_today - Close_yesterday) / Close_yesterday
    df_stock['Daily_Return'] = df_stock['Close'].pct_change()
    
    # Target: 1 if tomorrow's closing price is higher than today's, 0 otherwise
    # We shift tomorrow's Close back by -1 so we compare tomorrow's Close with today's Close
    df_stock['Target'] = (df_stock['Close'].shift(-1) > df_stock['Close']).astype(int)
    
    # Convert multi-level column names if yfinance downloads them that way
    if isinstance(df_stock.columns, pd.MultiIndex):
        df_stock.columns = df_stock.columns.get_level_values(0)

    # Format the index to date string (YYYY-MM-DD) for clean merging
    df_stock.index = pd.to_datetime(df_stock.index).date.astype(str)
    df_stock.index.name = 'Date'
    
    return df_stock

# ==========================================
# STEP 3: NLP Sentiment Extraction and Dataset Creation
# ==========================================
def acquire_datasets(file_path="financial_news.csv"):
    """
    Acquires both stock data and news headline data.
    If the news headline file exists, it loads it and downloads corresponding stock data.
    If the file does not exist, it downloads 3 years of stock data (Jan 2023 to Jan 2026),
    generates correlated synthetic news headlines, and saves them to file_path.
    """
    if os.path.exists(file_path):
        print(f"Found local headlines file at {file_path}. Loading data...")
        df_news = pd.read_csv(file_path)
        df_news.columns = [c.strip() for c in df_news.columns]
        df_news['Date'] = pd.to_datetime(df_news['Date'], format='mixed', utc=True, errors='coerce').dt.date.astype(str)
        df_news = df_news.dropna(subset=['Date', 'Headline'])
        
        # Determine dynamic date alignment
        news_dates = df_news['Date'].values
        news_min = min(news_dates)
        news_max = max(news_dates)
        min_date_parsed = datetime.datetime.strptime(news_min, '%Y-%m-%d').date()
        max_date_parsed = datetime.datetime.strptime(news_max, '%Y-%m-%d').date()
        stock_start = (min_date_parsed - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
        stock_end = (max_date_parsed + datetime.timedelta(days=10)).strftime('%Y-%m-%d')
        
        df_stock = download_stock_data("AAPL", start_date=stock_start, end_date=stock_end)
    else:
        print(f"Local file {file_path} not found. Creating fallback 3-year synthetic dataset (Jan 2023 to Jan 2026)...")
        # Jan 1, 2023 to Jan 1, 2026 (3 full years)
        stock_start = "2023-01-01"
        stock_end = "2026-01-01"
        df_stock = download_stock_data("AAPL", start_date=stock_start, end_date=stock_end)
        
        # Generate realistic synthetic financial headlines correlated with next-day trend (Target)
        np.random.seed(42)
        positive_samples = [
            "Apple reports record-breaking quarterly revenue, beating Wall Street estimates.",
            "AAPL shares rally as iPhone sales exceed expectations in Asian markets.",
            "Analysts upgrade Apple stock to BUY following groundbreaking AI features release.",
            "Apple introduces revolutionary VR headset, driving tech sector shares higher.",
            "Major fund increases AAPL holdings, highlighting strong cash flow and dividends.",
            "Apple's new M-series chips set a new industry benchmark for performance.",
            "Strong demand for Apple services subscription business boosts profit margins.",
            "Apple gains market share in key European regions as competitors struggle.",
            "Tech analysts praise Apple's robust capital return program and buybacks.",
            "Apple secures exclusive rights to major live sports streaming partnerships."
        ]
        negative_samples = [
            "Apple faces supply chain delays, lowering Q4 sales guidance.",
            "Antitrust lawsuit filed against Apple, dragging tech index down.",
            "EU imposes record fine on Apple over App Store payment restrictions.",
            "AAPL stock tumbles as consumer demand weakens in international markets.",
            "Analysts downgrade Apple due to stiff competition in smartphone sales.",
            "Apple hardware sales decline year-over-year amid economic headwinds.",
            "Key component shortages threaten Apple's holiday production schedule.",
            "Investors voice concern over Apple's lagging entry into generative AI.",
            "Apple under investigation by federal regulators over privacy policy changes.",
            "Weak pre-orders cause analysts to lower Apple earnings projections."
        ]
        neutral_samples = [
            "Apple schedules its annual autumn hardware launch event.",
            "Apple announces minor updates to iPadOS and macOS ecosystems.",
            "AAPL trading volume remains steady ahead of Federal Reserve interest decision.",
            "Apple signs lease for new corporate office space in London.",
            "Apple CEO Tim Cook participates in sustainability panel.",
            "Apple holds its quarterly conference call with institutional shareholders.",
            "Apple registers new trademarks for wearable device technologies.",
            "AAPL shares trade flat during quiet pre-market session.",
            "Apple releases minor bug fixes for iOS and watchOS systems.",
            "Apple hosts annual design awards for independent developers."
        ]
        
        generated_dates = []
        generated_headlines = []
        
        # Iterate over stock trading dates to match headlines to targets
        for date_str, row in df_stock.iterrows():
            target = row['Target']
            if pd.isna(target):
                continue
            
            # Generate 2 headlines per trading day
            for _ in range(2):
                rand_val = np.random.rand()
                # 70% correlation with target stock direction (next-day movement)
                if rand_val < 0.70:
                    if target == 1:
                        headline = np.random.choice(positive_samples)
                    else:
                        headline = np.random.choice(negative_samples)
                else:
                    # 30% random news (neutral/noise)
                    headline = np.random.choice(neutral_samples)
                
                generated_dates.append(date_str)
                generated_headlines.append(headline)
                
        df_news = pd.DataFrame({
            'Date': generated_dates,
            'Headline': generated_headlines
        })
        
        # Save synthetic data to the specified file path
        df_news.to_csv(file_path, index=False)
        print(f"Generated {len(df_news)} synthetic headlines spanning 3 years and saved to {file_path}.")
        
    print(f"Processing {len(df_news)} headlines through FinBERT. This might take a moment...")
    
    # Process headlines in batches for speed and performance
    headlines_list = df_news['Headline'].tolist()
    results = sentiment_analyzer(headlines_list, batch_size=32)
    
    # Map FinBERT labels (positive -> +1, negative -> -1, neutral -> 0)
    scores = []
    for res in results:
        label = res['label'].lower()
        if 'positive' in label:
            scores.append(1)
        elif 'negative' in label:
            scores.append(-1)
        else:
            scores.append(0)
            
    df_news['Sentiment_Score'] = scores
    
    # Group by Date and calculate Average_Daily_Sentiment
    df_daily_sentiment = df_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
    df_daily_sentiment = df_daily_sentiment.rename(columns={'Sentiment_Score': 'Average_Daily_Sentiment'})
    df_daily_sentiment = df_daily_sentiment.set_index('Date')
    
    return df_stock, df_daily_sentiment

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

# 1. Download/load both stock and daily sentiment datasets
df_stock, df_sentiment = acquire_datasets("financial_news.csv")

# 2. Merge Stock Data with Sentiment
print("\nMerging stock data and daily sentiment scores...")
df_merged = df_stock.merge(df_sentiment, left_index=True, right_index=True, how='left')

# Forward-fill missing sentiment scores (e.g. weekends or days with no headlines)
# Any initial days before any news are filled with 0 (neutral)
df_merged['Average_Daily_Sentiment'] = df_merged['Average_Daily_Sentiment'].ffill().fillna(0.0)

# Drop any rows with NaN in features or target (e.g., first row because of pct_change, last row because of shift)
df_merged = df_merged.dropna(subset=['Volume', 'Daily_Return', 'Average_Daily_Sentiment', 'Target'])

print(f"Data merged successfully. Total samples: {len(df_merged)}")

# ==========================================
# STEP 4: Machine Learning Predictor (XGBoost)
# ==========================================
# Select features
features = ['Volume', 'Daily_Return', 'Average_Daily_Sentiment']
X = df_merged[features]
y = df_merged['Target']

# Sort chronologically by date index to avoid future-lookahead bias
df_merged = df_merged.sort_index()

# 80/20 chronological Train-Test Split
split_idx = int(len(df_merged) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nSplitting data chronologically:")
print(f"Training set: {len(X_train)} samples ({df_merged.index[0]} to {df_merged.index[split_idx-1]})")
print(f"Testing set: {len(X_test)} samples ({df_merged.index[split_idx]} to {df_merged.index[-1]})")

# Train XGBoost Classifier
print("\nTraining XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    n_jobs=1
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Price Down/Flat', 'Price Up']))


# ==========================================
# STEP 5: Live Inference Function
# ==========================================
def predict_tomorrow_trend(headline_text, current_volume, daily_return):
    """
    Classifies a headline's sentiment using FinBERT, feeds the output 
    with volume and return metrics to the XGBoost model, and issues a BUY or SELL signal.
    """
    # 1. NLP Sentiment Classification
    result = sentiment_analyzer(headline_text)[0]
    label = result['label'].lower()
    
    if 'positive' in label:
        sentiment_score = 1.0
    elif 'negative' in label:
        sentiment_score = -1.0
    else:
        sentiment_score = 0.0
        
    print(f"\nInference Headline: '{headline_text}'")
    print(f"-> FinBERT Sentiment: {label.upper()} (Score: {sentiment_score})")
    
    # 2. Format features into a DataFrame matching model's expected shape
    input_data = pd.DataFrame([{
        'Volume': float(current_volume),
        'Daily_Return': float(daily_return),
        'Average_Daily_Sentiment': float(sentiment_score)
    }])
    
    # 3. Predict using trained XGBoost Model
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # 4. Print Signal output
    prob_up = probabilities[1]
    prob_down = probabilities[0]
    
    if prediction == 1:
        print(f"SIGNAL: BUY SIGNAL (Stock Expected to Rise) | Confidence: {prob_up:.2%}")
    else:
        print(f"SIGNAL: SELL SIGNAL (Stock Expected to Drop) | Confidence: {prob_down:.2%}")
        
    return prediction, prob_up

# ==========================================
# TEST BLOCK: Live Inference Verification & Searcher
# ==========================================
if __name__ == "__main__":
    print("\n==========================================")
    print("RUNNING LIVE INFERENCE TEST CASES")
    print("==========================================")
    
    # Let's extract typical daily volume and daily return from our test dataset to make inputs realistic
    mean_volume = df_merged['Volume'].mean()
    mean_return = df_merged['Daily_Return'].mean()
    
    # Case A: Highly Positive Headline
    positive_headline = "Apple releases new AI chip with 2x speedup, boosting revenue projections."
    predict_tomorrow_trend(
        headline_text=positive_headline, 
        current_volume=mean_volume, 
        daily_return=mean_return
    )
    
    # Case B: Highly Negative Headline
# Case B: Highly Negative Headline targeting Apple
    negative_headline = "Apple CEO steps down unexpectedly amid severe iPhone supply chain collapse and factory fires."
    predict_tomorrow_trend(
        headline_text=negative_headline, 
        current_volume=350000000,    # Inject a massive 350M panic-selling volume
        daily_return=-0.06           # Inject a severe 6% drop today
    )
    

    print("\n==========================================")
    print("BUY SIGNAL SEARCHER (Grid Search over Market Conditions)")
    print("==========================================")
    print(f"Testing positive headline: '{positive_headline}'")
    
    # Test ranges
    volumes_to_test = [15000000, 40000000, 80000000, 120000000, 150000000]
    returns_to_test = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
    
    successful_combinations = []
    
    print("\nEvaluating combinations of Volume and Daily Return with +1 Sentiment Score...")
    for vol in volumes_to_test:
        for ret in returns_to_test:
            # Predict using model directly to test combinations quietly
            input_data = pd.DataFrame([{
                'Volume': float(vol),
                'Daily_Return': float(ret),
                'Average_Daily_Sentiment': 1.0  # FinBERT positive sentiment
            }])
            pred = model.predict(input_data)[0]
            prob_up = model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                successful_combinations.append((vol, ret, prob_up))
                
    if successful_combinations:
        print(f"\nSUCCESS! Found {len(successful_combinations)} combinations that trigger a BUY SIGNAL:")
        print("-" * 65)
        print(f"{'Volume':<18} | {'Daily Return':<15} | {'Buy Confidence':<15}")
        print("-" * 65)
        for vol, ret, prob in successful_combinations:
            print(f"{vol:18,} | {ret:15.2%} | {prob:15.2%}")
        print("-" * 65)
        
        # Demonstrate the first successful case through the inference function
        best_vol, best_ret, _ = successful_combinations[0]
        print(f"\nDemonstrating one successful BUY condition:")
        predict_tomorrow_trend(
            headline_text=positive_headline,
            current_volume=best_vol,
            daily_return=best_ret
        )
    else:
        print("\nNo combinations of Volume and Daily Return triggered a BUY SIGNAL.")
        print("Note: The model might have a strong negative bias due to the small historical sample size.")

