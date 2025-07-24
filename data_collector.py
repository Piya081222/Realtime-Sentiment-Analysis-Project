import praw
import configparser
import json
import os
import sqlite3
from datetime import datetime

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

# --- 2. Set up Local Storage (SQLite) ---
db_path = os.path.join(os.path.dirname(__file__), 'reddit_comments.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS comments (
        id TEXT PRIMARY KEY,
        text TEXT,
        timestamp INTEGER,
        subreddit TEXT,
        score INTEGER
    )
''')
conn.commit()
print(f"SQLite database '{db_path}' ready.")

# --- 3. Stream Reddit Comments ---
print("Starting to stream comments from Reddit (r/all)...")
try:
    for comment in reddit.subreddit('all').stream.comments(skip_existing=True):
        try:
            comment_id = comment.id
            comment_text = comment.body
            comment_timestamp = int(comment.created_utc)
            subreddit_name = comment.subreddit.display_name
            comment_score = comment.score

            print(f" "
                  f"r/{subreddit_name} ({comment_score} upvotes): {comment_text[:100]}...")

            cursor.execute("INSERT OR IGNORE INTO comments (id, text, timestamp, subreddit, score) VALUES (?,?,?,?,?)",
                           (comment_id, comment_text, comment_timestamp, subreddit_name, comment_score))
            conn.commit()

        except Exception as e:
            print(f"Error processing comment: {e}")

except KeyboardInterrupt:
    print("\nStopping data collection.")
except Exception as e:
    print(f"An error occurred during streaming: {e}")
finally:
    conn.close()
    print("Database connection closed.")