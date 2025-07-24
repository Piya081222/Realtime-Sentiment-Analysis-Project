import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# --- 1. Download NLTK Data (Run Once) ---
# Why: NLTK needs specific datasets for stopwords, tokenization, and lemmatization.
# How: This block checks if the data is present and downloads it if not.
# You only need to run this successfully once on your machine.
# --- 1. Download NLTK Data (Run Once) ---
# Why: NLTK needs specific datasets for stopwords, tokenization, and lemmatization.
# How: This block checks if the data is present and downloads it if not.
# You only need to run this successfully once on your machine.
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError: # <--- THIS LINE IS CHANGED
    print("Downloading NLTK data (stopwords, punkt, wordnet)... This might take a moment.")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    print("NLTK data downloaded successfully.")

# Initialize NLTK tools after ensuring data is downloaded
stop_words = set(stopwords.words('english')) # Convert to a set for faster lookups
lemmatizer = WordNetLemmatizer()

# --- 2. Define the Preprocessing Function ---
# Why: To encapsulate all cleaning steps into one reusable function.
# How: Apply regex for noise removal, then tokenize, remove stopwords, and lemmatize.
def preprocess_text(text):
    # Ensure input is a string. If not, return empty string to avoid errors.
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase: Treats "Hello" and "hello" as the same.
    text = text.lower()

    # 2. Remove punctuation and numbers (Noise Removal):
    # re.sub() replaces patterns. '[^a-z\s]' means "anything NOT a lowercase letter or a space".
    # We replace those with an empty string.
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenization: Break the text into individual words.
    tokens = word_tokenize(text)

    # 4. Remove Stopwords and Lemmatize:
    # We iterate through each token.
    # If it's not a stop word, we lemmatize it (get its base form).
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]

    # 5. Join tokens back into a single string, separated by spaces.
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# --- 3. Load Data from SQLite and Process ---
# Why: Read the raw comments from our database and apply the cleaning function.
# How: Use pandas to read data, then apply the preprocess_text function to the 'text' column.
db_path = os.path.join(os.path.dirname(__file__), 'reddit_comments.db')
conn = sqlite3.connect(db_path)

try:
    # Read all comments from the 'comments' table into a pandas DataFrame.
    # A DataFrame is like a super-powered spreadsheet in Python, great for data manipulation.
    df = pd.read_sql_query("SELECT id, text, timestamp, subreddit, score FROM comments", conn)
    print(f"Loaded {len(df)} comments from database.")

    # Apply the preprocessing function to the 'text' column of our DataFrame.
    # The.apply() method runs our function on every value in that column.
    df['processed_text'] = df['text'].apply(preprocess_text)
    print("Text preprocessing complete.")

    # Display some processed data to see the effect of cleaning.
    print("\nOriginal vs. Processed Text (first 5 rows):")
    # We'll print the 'text' (original) and 'processed_text' columns side-by-side.
    print(df[['text', 'processed_text']].head())

    # IMPORTANT: For this phase, we are just displaying the processed text.
    # In the next phase (Sentiment Analysis), we will use this 'processed_text' directly.
    # We are NOT saving it back to the database in this script, to keep things modular.

except pd.io.sql.DatabaseError as e:
    print(f"Error reading from database: {e}. Make sure data_collector.py has run and created the 'comments' table and collected some data.")
    df = pd.DataFrame() # Return an empty DataFrame if there's an error
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Always close the database connection when done.
    conn.close()
    print("Database connection closed.")