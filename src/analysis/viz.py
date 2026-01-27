import streamlit as st
import pandas as pd
from collections import Counter
from wordcloud import WordCloud

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================
def generate_wordcloud(text):
    """Generate word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud.to_array()

def get_video_preview_thumbnails(video_id):
    """Get YouTube auto-generated thumbnails."""
    return {
        "beginning": f"https://img.youtube.com/vi/{video_id}/1.jpg",
        "middle": f"https://img.youtube.com/vi/{video_id}/2.jpg",
        "end": f"https://img.youtube.com/vi/{video_id}/3.jpg"
    }

def plot_top_comment_words(comments_df, n=15):
    """Plot top words from comments."""
    words = " ".join(comments_df["comment_text"].astype(str)).lower().split()
    words = [w for w in words if w.isalpha() and w not in STOPWORDS and len(w) > 2]
    top = Counter(words).most_common(n)
    if not top:
        st.info("Not enough relevant words to display.")
        return
    df = pd.DataFrame(top, columns=["word", "count"])
    st.bar_chart(df.set_index("word"))