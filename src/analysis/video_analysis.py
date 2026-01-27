import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.sentiment import SentimentIntensityAnalyzer

from collections import Counter
import pandas as pd
from api.youtube_api import get_channel_videos, get_video_details

sia = SentimentIntensityAnalyzer()

# =========================================================
# ADVERTISING-SPECIFIC FUNCTIONS
# =========================================================
def analyze_video_duration_for_ads(duration_seconds):
    """Analyze if video duration is optimal for advertising."""
    if duration_seconds < 6:
        return "⚡ Bumper Ad (6s) - Ideal for quick brand awareness", "success"
    elif duration_seconds <= 15:
        return "✅ Short format - Perfect for skippable ads", "success"
    elif duration_seconds <= 30:
        return "👍 Good duration - Optimal for TrueView ads", "success"
    elif duration_seconds <= 60:
        return "⚠️ Medium duration - May lose audience after 30s", "warning"
    else:
        minutes = duration_seconds // 60
        return f"❌ Too long ({minutes}min+) - High abandonment risk for ads", "error"

def analyze_title_for_ads(title):
    """Analyze title effectiveness for advertising."""
    score = 0
    recommendations = []
    
    if 40 <= len(title) <= 70:
        score += 25
        recommendations.append("✅ Optimal length (40-70 characters)")
    elif len(title) < 40:
        score += 10
        recommendations.append("⚠️ Short title - consider adding more context")
    else:
        recommendations.append("❌ Title too long - may be cut off on mobile")
    
    if re.search(r'\d+', title):
        score += 20
        recommendations.append("✅ Contains numbers (increases CTR)")
    else:
        recommendations.append("💡 Consider adding numbers for impact")
    
    power_words = ['free', 'new', 'best', 'top', 'how', 'why', 'ultimate', 'guide', 'tips']
    if any(word in title.lower() for word in power_words):
        score += 25
        recommendations.append("✅ Contains power words")
    else:
        recommendations.append("💡 Add action keywords")
    
    if title[0].isupper():
        score += 15
        recommendations.append("✅ First letter capitalized")
    
    if any(p in title for p in ['!', '?', ':', '|']):
        score += 15
        recommendations.append("✅ Uses effective punctuation")
    else:
        recommendations.append("💡 Consider punctuation for emphasis")
    
    return score, recommendations

def detect_cta_in_description(description):
    """Detect call-to-action elements in description."""
    cta_patterns = [
        r'(click|tap|visit|check out|learn more|subscribe|buy|shop|get|download)',
        r'(link in.*description|link below|in.*bio)',
        r'(www\.|https?://)',
    ]
    
    ctas_found = []
    for pattern in cta_patterns:
        matches = re.findall(pattern, description.lower())
        if matches:
            ctas_found.extend(matches)
    
    return len(set(ctas_found))

def analyze_engagement_velocity(views, likes, comments, days_old):
    """Analyze engagement velocity (daily metrics)."""
    if days_old == 0:
        days_old = 1
    
    views_per_day = views / days_old
    likes_per_day = likes / days_old
    comments_per_day = comments / days_old
    
    return {
        "views_per_day": views_per_day,
        "likes_per_day": likes_per_day,
        "comments_per_day": comments_per_day,
        "engagement_per_day": (likes + comments) / days_old
    }

def calculate_viral_coefficient(views, likes, comments, channel_avg_engagement):
    """Calculate virality coefficient compared to channel average."""
    video_engagement = (likes + comments) / max(views, 1)
    
    if channel_avg_engagement > 0:
        viral_score = (video_engagement / channel_avg_engagement) * 100
        
        if viral_score >= 150:
            return viral_score, "🚀 VIRAL - Exceeds channel average by 150%+", "success"
        elif viral_score >= 100:
            return viral_score, "📈 Excellent - Above average performance", "success"
        elif viral_score >= 75:
            return viral_score, "👍 Good - Near average performance", "info"
        else:
            return viral_score, "⚠️ Below average performance", "warning"
    
    return None, "⚠️ No channel data for comparison", "info"

def analyze_comment_sentiment_detailed(comments_df):
    """Detailed sentiment analysis of comments."""
    if comments_df.empty:
        return None
    
    comments_df["sentiment"] = comments_df["comment_text"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"]
    )
    
    positive = len(comments_df[comments_df["sentiment"] > 0.05])
    neutral = len(comments_df[(comments_df["sentiment"] >= -0.05) & (comments_df["sentiment"] <= 0.05)])
    negative = len(comments_df[comments_df["sentiment"] < -0.05])
    
    total = len(comments_df)
    
    return {
        "positive_pct": (positive / total) * 100 if total > 0 else 0,
        "neutral_pct": (neutral / total) * 100 if total > 0 else 0,
        "negative_pct": (negative / total) * 100 if total > 0 else 0,
        "avg_sentiment": comments_df["sentiment"].mean(),
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "total": total
    }

def extract_top_keywords(title, tags, description):
    """Extract top keywords from video content."""
    text = f"{title} {' '.join(tags)} {description[:200]}"
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    vectorizer = CountVectorizer(stop_words="english", max_features=15, ngram_range=(1, 2))
    try:
        X = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except:
        return []
    
# =========================================================
# EMOTION MAP & TOPIC MODELING
# =========================================================
def emotion_and_topics_from_comments(comments_df, n_topics=3, max_features=5000):
    """Advanced comment analysis."""
    res = {
        "sentiment_counts": None,
        "avg_sentiment": None,
        "repr_comments": {},
        "topics": [],
    }
    if comments_df is None or comments_df.empty:
        return res

    comments_df = comments_df.copy()
    comments_df["sentiment"] = comments_df["comment_text"].apply(
        lambda t: sia.polarity_scores(str(t))["compound"]
    )
    comments_df["sent_bucket"] = comments_df["sentiment"].apply(
        lambda s: "positive" if s > 0.2 else ("negative" if s < -0.2 else "neutral")
    )

    res["sentiment_counts"] = comments_df["sent_bucket"].value_counts().to_dict()
    res["avg_sentiment"] = float(comments_df["sentiment"].mean())

    for b in ["positive", "neutral", "negative"]:
        dfb = comments_df[comments_df["sent_bucket"] == b]
        if not dfb.empty:
            dfb_sorted = dfb.sort_values("like_count", ascending=False)
            res["repr_comments"][b] = dfb_sorted["comment_text"].head(5).tolist()
        else:
            res["repr_comments"][b] = []

    texts = comments_df["comment_text"].astype(str).tolist()
    if len(texts) >= 10:
        try:
            cv = CountVectorizer(stop_words="english", max_features=max_features)
            X = cv.fit_transform(texts)
            lda = LDA(n_components=min(n_topics, 6), random_state=0, learning_method="batch")
            lda.fit(X)
            words = cv.get_feature_names_out()
            topics = []
            for i, comp in enumerate(lda.components_):
                terms = [words[idx] for idx in comp.argsort()[-8:][::-1]]
                topics.append({"topic_id": i, "top_words": terms})
            res["topics"] = topics
        except Exception:
            res["topics"] = []
    else:
        res["topics"] = []

    return res

# =========================================================
# CHANNEL TAG HISTORY & LENGTH VS ENGAGEMENT
# =========================================================
def channel_tags_history(api_key, channel_id, max_videos=50):
    """Analyze channel tag usage and collect video data."""
    vids = get_channel_videos(api_key, channel_id, max_results=max_videos)
    tags_counter = Counter()
    data = []
    for vid in vids:
        v = get_video_details(api_key, vid)
        if v:
            data.append(v)
            tags_counter.update([t.lower() for t in v.get("tags", []) if isinstance(t, str)])
    return tags_counter, pd.DataFrame(data)