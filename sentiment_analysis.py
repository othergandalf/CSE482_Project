import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

#nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df = pd.read_csv("reddit_olympics_comments_with _valid_URLs_2023-2024.csv")
if 'Body' not in df.columns:
    raise KeyError("The CSV file does not contain a column named 'Body'.")

def get_sentiment_score(text):
    if pd.isnull(text):
        return None
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']

df['Sentiment'] = df['Body'].apply(get_sentiment_score)

df.to_csv("sentiment_analysised_olypics_data.csv", index=False)

print("The updated CSV file with sentiment scores has been saved as 'output_with_sentiment.csv'.")