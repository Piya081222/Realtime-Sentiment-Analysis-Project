import praw
import configparser
import os
import sqlite3
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline # Our AI power-up!

# --- 1. Load Credentials ---
# Why: Securely load your Reddit API keys and login details.
# How: configparser helps read the .cfg file we created.
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'secrets', 'credentials.cfg')
config.read(config_path)

reddit = praw.Reddit(
    client_id=config['reddit']['client_id'],
    client_secret=config['reddit']['client_secret'],
    password=config['reddit']['password'],
    user_agent=config['reddit']['user_agent'],
    username=config['reddit']['username'],
)
print("Reddit instance initialized successfully.")

# --- 2. Set up Local Storage (SQLite) for processed data ---
# Why: Store the incoming real-time data, along with its sentiment, persistently on your laptop.
# We'll use a *new* database file to keep raw and processed data separate for clarity.
# How: Connect to a database file and create a table if it doesn't exist.
db_path = os.path.join(os.path.dirname(__file__), 'reddit_sentiment_data.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS comments_sentiment (
        id TEXT PRIMARY KEY,
        original_text TEXT,
        processed_text TEXT,
        timestamp INTEGER,
        subreddit TEXT,
        score INTEGER,
        sentiment_label TEXT,  -- To store 'POSITIVE' or 'NEGATIVE'
        sentiment_score REAL   -- To store the confidence score (e.g., 0.99)
    )
''')
conn.commit()
print(f"SQLite database '{db_path}' for sentiment data ready.")

# --- 3. NLTK Preprocessing Setup ---
# Why: To clean and normalize text before feeding it to the AI model.
# How: Ensure NLTK data is downloaded and define the preprocessing function.
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    print("NLTK data is already available.")
except LookupError:
    print("Downloading NLTK data (stopwords, punkt, wordnet)... This might take a moment.")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded successfully.")

# Initialize NLTK tools after ensuring data is downloaded
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    tokens = word_tokenize(text)
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# --- 4. Load Hugging Face Sentiment Analysis Pipeline ---
# Why: To use a powerful, pre-trained AI model for sentiment classification.
# How: Use the transformers library's pipeline function with a specific model.
print("Loading Hugging Face sentiment analysis model... This will download the model the first time.")
# You can use the default or a specific one like 'siebert/sentiment-roberta-large-english'
# Let's stick with the default for now as it's less resource-intensive.
sentiment_analyzer = pipeline("sentiment-analysis") 
print("Sentiment analysis model loaded.")

# --- 5. Stream, Process, Analyze, and Store ---
# Why: The main loop to continuously fetch comments, clean them, get sentiment, and save.
# How: Use PRAW's streaming, call preprocessing, call the AI pipeline, and insert into SQLite.
print("Starting real-time sentiment analysis of Reddit comments...")
try:
    # reddit.subreddit('all').stream.comments() continuously fetches new comments.
    # skip_existing=True ensures we only get new comments from when the script starts.
    for comment in reddit.subreddit('all').stream.comments(skip_existing=True):
        try:
            comment_id = comment.id
            original_text = comment.body
            comment_timestamp = int(comment.created_utc) # UTC timestamp
            subreddit_name = comment.subreddit.display_name
            comment_score = comment.score

            # Preprocess the text using our defined function
            processed_text = preprocess_text(original_text)

            # Perform sentiment analysis using the Hugging Face model
            if processed_text: # Only analyze if there's meaningful text after preprocessing
                sentiment_result = sentiment_analyzer(processed_text)
                
                # --- FIX APPLIED HERE ---
                # The result is a LIST containing a DICTIONARY, e.g., [{'label': '...', 'score': ...}]
                # We need to check if the list is not empty and then access the first element.
                if sentiment_result and isinstance(sentiment_result, list):
                    sentiment_dict = sentiment_result[0] # Get the dictionary from the list
                    sentiment_label = sentiment_dict['label']
                    sentiment_score = sentiment_dict['score']
                else:
                    # Fallback if the model returns an unexpected format
                    sentiment_label = "NEUTRAL"
                    sentiment_score = 0.5
            else:
                # Handle cases where text becomes empty after preprocessing (e.g., just punctuation or stopwords)
                sentiment_label = "NEUTRAL" # Assign a neutral label
                sentiment_score = 0.5       # Assign a default neutral score

            # Print to console for real-time feedback
            print(f"r/{subreddit_name} | Sentiment: {sentiment_label} ({sentiment_score:.2f}) | "
                  f"Text: {original_text[:70].replace(chr(10), ' ')}...") # Show first 70 chars, replace newlines

            # Store all relevant data in our new SQLite database table
            cursor.execute("INSERT OR IGNORE INTO comments_sentiment (id, original_text, processed_text, timestamp, subreddit, score, sentiment_label, sentiment_score) VALUES (?,?,?,?,?,?,?,?)",
                           (comment_id, original_text, processed_text, comment_timestamp, subreddit_name, comment_score, sentiment_label, sentiment_score))
            conn.commit() # Save each comment immediately

        except praw.exceptions.PRAWException as e:
            # Catch potential Reddit API issues (e.g., deleted comment)
            print(f"A PRAW-related error occurred for comment '{comment.id}': {e}")
            continue
        except Exception as e:
            # Log the error but continue streaming if an individual comment fails
            # We check if comment object exists before trying to get its id
            c_id = comment.id if 'comment' in locals() and hasattr(comment, 'id') else 'unknown'
            print(f"Error processing comment '{c_id}': {e}")
            continue # Continue to the next comment in the stream

except KeyboardInterrupt:
    print("\nStopping real-time sentiment analysis.")
except Exception as e:
    print(f"An unexpected error occurred during streaming: {e}")
finally:
    # Always close the database connection when the script stops
    if 'conn' in locals() and conn:
        conn.close()
        print("Database connection closed.")