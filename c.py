import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load CSV file (replace with your actual file path)
df = pd.read_csv(r"C:\Users\vaish\Downloads\amazon.csv\meesho Orders Aug.csv")

# Ensure correct column name
text_column = "text_column_name"  # Change this to match your CSV column name

# Download VADER lexicon
nltk.download('vader_lexicon')


