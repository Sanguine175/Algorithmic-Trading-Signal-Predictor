# Algorithmic Trading Signal Predictor

A complete machine learning and natural language processing (NLP) pipeline that predicts daily stock movement trends for Apple (`AAPL`) by combining historical stock metrics with financial news sentiment analysis.

## Features

1. **Stock Data Acquisition:** Downloads historical stock price data using `yfinance`, calculates daily return percentages, and defines binary targets (1 for price rise, 0 for flat/fall next day).
2. **NLP Sentiment Extraction:** Loads the `ProsusAI/finbert` model via HuggingFace's `transformers` pipeline to analyze financial headlines and groups them to calculate daily average sentiment scores.
3. **Data Merging & Feature Engineering:** Merges daily stock data with news sentiment scores using forward-filling to handle weekends and days without news.
4. **Machine Learning Classifier (XGBoost):** Splits dataset chronologically (80/20 train/test) to avoid time-series leakage, and trains an XGBoost model on `Volume`, `Daily_Return`, and `Average_Daily_Sentiment`.
5. **Live Inference Function:** A standalone function (`predict_tomorrow_trend`) that parses a custom news headline, merges it with current market volume and return metrics, and issues a clear **BUY SIGNAL** or **SELL SIGNAL** with a confidence score.
6. **Buy Signal Searcher:** Implements a grid search over various Volume and Daily Return parameters to isolate exactly which market conditions trigger positive sentiment into a BUY condition.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sanguine175/Algorithmic-Trading-Signal-Predictor.git
   cd Algorithmic-Trading-Signal-Predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the main pipeline using python:
```bash
python trading_pipeline.py
```

---

## File Structure
* `trading_pipeline.py`: The core pipeline executing data download, FinBERT sentiment parsing, XGBoost training, validation metrics, and live inference test cases.
* `requirements.txt`: Defines libraries required (`yfinance`, `transformers`, `xgboost`, `pandas`, `scikit-learn`, `torch`).
* `financial_news.csv`: Preprocessed 3-year news headline dataset mapped to AAPL stock target direction.
* `preprocess.py`: Preprocessing utility to filter Apple news from raw analyst ratings.
