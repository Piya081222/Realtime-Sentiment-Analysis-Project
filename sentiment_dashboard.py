import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Reddit Sentiment Dashboard")
st.title("🧠 Real-Time Reddit Sentiment Dashboard")

# --- Auto-Refresh Every 15 Seconds ---
st_autorefresh(interval=15000, key="auto_refresh")

# --- Load Data Function ---
@st.cache_data(ttl=10)
def load_data():
    db_path = os.path.join(os.path.dirname(__file__), 'reddit_sentiment_data.db')
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM comments_sentiment ORDER BY timestamp DESC LIMIT 1000", conn
        )
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    finally:
        conn.close()

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Comments")

time_filter = st.sidebar.selectbox(
    "Time Range:",
    ["Last 1 Hour", "Last 6 Hours", "Last 12 Hours", "Last 24 Hours"],
    index=1
)
time_threshold = {
    "Last 1 Hour": datetime.utcnow() - timedelta(hours=1),
    "Last 6 Hours": datetime.utcnow() - timedelta(hours=6),
    "Last 12 Hours": datetime.utcnow() - timedelta(hours=12),
    "Last 24 Hours": datetime.utcnow() - timedelta(hours=24),
}[time_filter]

if not df.empty:
    df = df[df['datetime'] >= time_threshold]

all_subs = sorted(df['subreddit'].unique())
selected_subs = st.sidebar.multiselect("Subreddits:", all_subs, default=all_subs)
df = df[df['subreddit'].isin(selected_subs)]

sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
selected_sentiments = st.sidebar.multiselect("Sentiments:", sentiments, default=sentiments)
df = df[df['sentiment_label'].isin(selected_sentiments)]

# --- Status Info ---
st.markdown(f"**Last Updated:** `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC`")
st.markdown(f"**Showing {len(df)} comments** after applying filters.")

st.divider()

# --- Layout Columns ---
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

if not df.empty:
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment_label'].value_counts()
        fig_sentiment = px.pie(
            sentiment_counts,
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'POSITIVE': '#2ca02c',
                'NEGATIVE': '#d62728',
                'NEUTRAL': '#1f77b4'
            }
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        st.subheader("Sentiment Over Time")
        df['hour'] = df['datetime'].dt.floor('h')
        hourly_sentiment = df.groupby('hour')['adjusted_score'].mean().reset_index()
        fig_time = px.line(hourly_sentiment, x='hour', y='adjusted_score',
                           labels={'hour': 'Time', 'adjusted_score': 'Avg Sentiment'})
        st.plotly_chart(fig_time, use_container_width=True)

    subreddit_counts = df['subreddit'].value_counts()
    frequent_subs = subreddit_counts[subreddit_counts >= 5].index

    if not frequent_subs.empty:
        sentiment_by_sub = (
            df[df['subreddit'].isin(frequent_subs)]
            .groupby('subreddit')['adjusted_score']
            .mean()
            .sort_values()
        )

        with col3:
            st.subheader("Most Negative Subreddits")
            fig_neg = px.bar(
                sentiment_by_sub.head(10),
                x=sentiment_by_sub.head(10).values,
                y=sentiment_by_sub.head(10).index,
                orientation='h',
                labels={'x': 'Avg Sentiment Score', 'y': 'Subreddit'}
            )
            st.plotly_chart(fig_neg, use_container_width=True)

        with col4:
            st.subheader("Most Positive Subreddits")
            top_sorted = sentiment_by_sub.tail(10).sort_values(ascending=False)
            fig_pos = px.bar(
                top_sorted,
                x=top_sorted.values,
                y=top_sorted.index,
                orientation='h',
                labels={'x': 'Avg Sentiment Score', 'y': 'Subreddit'}
            )
            st.plotly_chart(fig_pos, use_container_width=True)

    # --- Latest Comments Table ---
    st.header("🗨️ Latest Comments")
    comment_df = df[['datetime', 'subreddit', 'original_text', 'sentiment_label', 'adjusted_score']].copy()
    comment_df['datetime'] = comment_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    comment_df['adjusted_score'] = comment_df['adjusted_score'].round(2)

    # Apply coloring to sentiment label
    def style_sentiment(val):
        color = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}.get(val, 'black')
        return f'color: {color}; font-weight: bold'

    styled_df = comment_df.style.applymap(style_sentiment, subset=['sentiment_label'])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=350,
        hide_index=True
    )

else:
    st.warning("⚠️ No data available for selected filters or time range.")
