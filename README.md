# Olympic Sentiment Analysis: Reddit and Economic Indicators

## Overview
This project analyzes the relationship between Reddit sentiment and economic indicators in France during the lead-up to the 2024 Olympics. The analysis combines Reddit comments from r/france and r/olympics with various French economic indicators to explore correlations between public sentiment and economic trends.

## Data Sources
- **Reddit Data**: Comments and posts from r/france and r/olympics (May-November 2024)
- **Economic Indicators**: FRED economic data including:
  - Air Transport Price
  - Combined Transport Price
  - CPI Growth
  - EU Policy Uncertainty
  - Unemployment Rate
  - French Stock Market Performance
  - Tourism Job Index

## Methodology

### Sentiment Analysis
Used VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment scoring:
```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    if pd.isnull(text):
        return None
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']
```

### Data Processing
The analysis was conducted at two levels:

1. **Daily Analysis**
```python
def prepare_economic_data(daily_df):
    daily_df['Tourism_Jobs_Change'] = daily_df['Tourism_Jobs_Index'].diff()
    daily_df['Tourism_Jobs_Direction'] = np.where(
        daily_df['Tourism_Jobs_Change'] > 0, 1,
        np.where(daily_df['Tourism_Jobs_Change'] < 0, -1, 0)
    )
    return daily_df
```

2. **Monthly Analysis**
```python
def prepare_monthly_reddit_data(reddit_df):
    reddit_df['month_date'] = pd.to_datetime(reddit_df['Created']).dt.to_period('M')
    monthly_sentiment = reddit_df.groupby('month_date').agg(
        mean_sentiment=('Sentiment', 'mean'),
        post_count=('Sentiment', 'count')
    )
    return monthly_sentiment
```

## Key Findings

### Daily Analysis
- Pearson correlation coefficient: 0.114
- T-test p-value: 0.004
- Direction matching: 41.1%

### Monthly Analysis
Correlations with sentiment:
- All Shares Total France: 0.620
- Unemployment Rate: 0.574
- EU Policy Uncertainty: -0.211
- Air Transport Price: -0.126
- Combined Transport Price: 0.090

## Conclusions
The analysis reveals stronger correlations between Reddit sentiment and broader economic indicators at the monthly level compared to daily fluctuations. The strongest correlation was found with the French stock market performance (r = 0.620), while consumer-facing metrics showed weaker correlations. This suggests that Olympic-related public sentiment aligns more closely with longer-term economic trends rather than daily economic fluctuations.

## Dependencies
- pandas
- numpy
- nltk (with vader_lexicon)
- matplotlib
- seaborn
- scipy

## Installation
```bash
pip install pandas numpy nltk matplotlib seaborn scipy
python -m nltk.downloader vader_lexicon
```

## Data Collection Note
Data was collected using Reddit's API during the 2024 transition period following the 2023 API pricing changes. The dataset includes comments and posts from both English and French language subreddits, providing a diverse sampling of public discourse around the Olympic preparations.

## Citation
If you use this analysis, please cite:

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

## License
[Insert your chosen license here]
