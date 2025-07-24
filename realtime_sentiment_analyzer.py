import praw
import configparser
import os
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# --- 1. Load Credentials ---
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

# --- 2. Set up SQLite Database (with adjusted_score) ---
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
        sentiment_label TEXT,
        sentiment_score REAL,
        entities TEXT,
        adjusted_score REAL  -- NEW COLUMN for intuitive score
    )
''')
conn.commit()
print(f"SQLite database ready with 'adjusted_score' column.")

# --- 3. NLTK Preprocessing Setup ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download(['stopwords', 'punkt', 'wordnet'], quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stop_words])

# --- 4. Load Hugging Face Pipelines ---
print("Loading Hugging Face models...")
sentiment_analyzer = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner", grouped_entities=True)
print("Sentiment and NER models loaded.")

# --- 5. Main Streaming Logic with Fixes ---
print("Starting real-time analysis...")
BATCH_SIZE = 50
comment_batch = []

def write_batch_to_db(batch):
    if not batch: return
    try:
        cursor.executemany("INSERT OR IGNORE INTO comments_sentiment VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
        conn.commit()
        print(f"--- Wrote batch of {len(batch)} comments to DB ---")
    except Exception as e:
        print(f"--- DATABASE BATCH WRITE FAILED: {e} ---")

try:
    for comment in reddit.subreddit('all').stream.comments(skip_existing=True):
        try:
            original_text = comment.body
            processed_text = preprocess_text(original_text)
            
            # Default values
            sentiment_label, sentiment_score, adjusted_score, entities_str = "NEUTRAL", 0.5, 0.0, "None"

            # FIXED: Handle long comments and calculate adjusted_score
            if processed_text:
                try:
                    # Run analysis
                    sentiment_result = sentiment_analyzer(processed_text)
                    ner_results = ner_pipeline(original_text)

                    # Process sentiment
                    sentiment_label = sentiment_result[0]['label']
                    sentiment_score = sentiment_result[0]['score']
                    
                    # Create the new adjusted_score
                    if sentiment_label == 'POSITIVE':
                        adjusted_score = sentiment_score
                    elif sentiment_label == 'NEGATIVE':
                        adjusted_score = -sentiment_score # Make it negative
                    else:
                        adjusted_score = 0.0

                    # Process entities
                    entities_list = [f"{entity['entity_group']}: {entity['word']}" for entity in ner_results]
                    entities_str = ", ".join(entities_list) if entities_list else "None"
                
                except Exception as model_error:
                    # This will catch errors from comments that are too long
                    print(f"Skipping comment {comment.id} due to model error: {model_error}")
                    continue # Skip to the next comment

            # Add to batch
            comment_data = (comment.id, original_text, processed_text, int(comment.created_utc),
                            comment.subreddit.display_name, comment.score, sentiment_label,
                            sentiment_score, entities_str, adjusted_score)
            comment_batch.append(comment_data)

            if len(comment_batch) >= BATCH_SIZE:
                write_batch_to_db(comment_batch)
                comment_batch = []

        except Exception as e:
            print(f"Error processing comment '{comment.id}': {e}")
            continue

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    write_batch_to_db(comment_batch)
    if 'conn' in locals() and conn:
        conn.close()
        print("Database connection closed.")