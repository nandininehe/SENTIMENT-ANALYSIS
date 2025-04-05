import pandas as pd
import nltk
import os
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ✅ Update with the correct file path
file_path =""  # Update this!

# ✅ Check if file exists
if not os.path.exists(r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv"):
    print(f"❌ Error: File not found at {r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv"}")
    exit()

# ✅ Load the CSV file
df = pd.read_csv(r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv")
print("✅ CSV file loaded successfully!")

# ✅ Use the correct column for sentiment analysis
text_column = "Reason for Credit Entry"  # Update based on your data!

if text_column not in df.columns:
    print(f"❌ Error: Column '{text_column}' not found in the CSV file!")
    print("🔍 Available columns:", df.columns)
    exit()

# ✅ Download VADER lexicon
nltk.download('vader_lexicon')

# ✅ Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# ✅ Function for VADER Sentiment Analysis
def get_vader_sentiment(text):
    scores = sia.polarity_scores(str(text))  # Convert to string for safety
    return "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"

df["VADER Sentiment"] = df[text_column].apply(get_vader_sentiment)

# ✅ Function for TextBlob Sentiment Analysis
def get_textblob_sentiment(text):
    analysis = TextBlob(str(text))
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

df["TextBlob Sentiment"] = df[text_column].apply(get_textblob_sentiment)

# ✅ Save results
output_file = r"C:\Users\vaish\Documents\sentiment_results.csv"
df.to_csv(output_file, index=False)
print(f"✅ Sentiment analysis completed and saved as {output_file}")

# ✅ Display sentiment summary
print("\n📊 VADER Sentiment Summary:\n", df["VADER Sentiment"].value_counts())
print("\n📊 TextBlob Sentiment Summary:\n", df["TextBlob Sentiment"].value_counts())

# ✅ Plot Sentiment Distribution
plt.figure(figsize=(6, 4))
df["VADER Sentiment"].value_counts().plot(kind="bar", color=["green", "red", "gray"])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Analysis Results")
plt.show()
