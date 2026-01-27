import os
import re
import sys
import time
import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from dateutil import parser as dtparser

import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Optional libraries for thumbnail analysis
try:
    from PIL import Image, ImageStat
except Exception:
    Image = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# Visualization
import altair as alt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)


from utils.helpers import (
    extract_video_id,
    iso8601_to_hms,
    iso8601_to_seconds,
    clean_html,
    days_since
)

from api.youtube_api import (
    get_video_details,
    fetch_comments,
    get_channel_videos,
    get_channel_engagement,
    plot_channel_history
)

from analysis.video_analysis import (
    analyze_video_duration_for_ads,
    analyze_title_for_ads,
    detect_cta_in_description,
    analyze_engagement_velocity,
    calculate_viral_coefficient,
    analyze_comment_sentiment_detailed,
    extract_top_keywords,
    emotion_and_topics_from_comments,
    channel_tags_history
)

from analysis.viz import (
    generate_wordcloud,
    get_video_preview_thumbnails,
    plot_top_comment_words
)

from storage.sheets import (
    load_portfolio_from_sheet,
    append_to_sheet
)

from analysis.thumbnail import analyze_thumbnail

def get_secret(name: str, default=None):
    # 1) Prefer environment variables (Docker Compose / .env)
    val = os.getenv(name)
    if val not in (None, ""):
        return val

    # 2) Try Streamlit secrets only if available (doesn't crash if missing)
    try:
        return st.secrets[name]
    except Exception:
        return default
    

# =========================================================
# API KEY
# =========================================================
api_key = get_secret("YT_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("🔑 YouTube API Key", type="password")

if not api_key:
    st.error("Missing YouTube API key. Set YT_API_KEY in environment variables (Docker/.env) or Streamlit secrets.")
    st.stop()


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="YouTube Ad Analyzer Pro", layout="wide")
st.title("📊 YouTube Ad Analyzer – Marketing Insights Pro")
st.caption("Analyze video performance, audience sentiment, and channel history with advanced AI-powered insights.")

# =========================================================
# NLTK SETUP
# =========================================================
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

try:
    nltk.data.find("corpora/stopwords.zip")
except LookupError:
    nltk.download("stopwords")

sia = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words("english"))


# =========================================================
# SESSION STATE
# =========================================================
if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = load_portfolio_from_sheet()

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None

# Añadido: Control para evitar re-análisis automático
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "comparison_done" not in st.session_state:
    st.session_state["comparison_done"] = False


# =========================================================
# UI TABS
# =========================================================
tab1, tab2 = st.tabs(["🎥 Video Analysis", "📊 Dashboard"])

# =========================================================
# TAB 1: VIDEO ANALYSIS
# =========================================================
with tab1:
    st.header("🔧 Analysis Inputs")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        video_url = st.text_input("🎥 Main Video URL",
                                  placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
    with c2:
        channel_id = st.text_input("📺 Channel ID (optional)",
                                   placeholder="UC_x5XG1OV2P6uZZ5FSM9Ttw")
    with c3:
        video_url_2 = st.text_input("🎥 Comparison Video URL",
                                    placeholder="https://www.youtube.com/watch?v=VIDEO_ID_2")

    max_comments = st.slider("Comments to download (for sentiment analysis)", 0, 500, 100)

    colb1, colb2 = st.columns(2)
    with colb1:
        run = st.button("🚀 Analyze Main Video", type="primary")
    with colb2:
        compare = st.button("🆚 Compare Two Videos")

    st.markdown("---")

    # ========== MAIN VIDEO ANALYSIS ==========
    if run:
        st.session_state["analysis_done"] = True
        st.session_state["comparison_done"] = False

        vid = extract_video_id(video_url)
        if not vid:
            st.error("❌ Invalid video URL.")
            st.stop()

        with st.spinner("🔍 Analyzing video..."):
            v = get_video_details(api_key, vid)
            if not v:
                st.error("❌ Could not retrieve video information.")
                st.stop()

            engagement_rate = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
            duration_seconds = iso8601_to_seconds(v["duration_iso"])
            days_old = days_since(v["published_at"])
            
            st.session_state["last_analysis"] = {
                "video_id": v["video_id"],
                "title": v["title"],
                "channel": v["channel_title"],
                "engagement_rate": engagement_rate,
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

            # ===== VIDEO HEADER =====
            col1, col2 = st.columns([1, 2])
            with col1:
                if v.get("thumbnail"):
                    st.image(v["thumbnail"], width=400)
            with col2:
                st.subheader(v["title"])
                st.write(f"**Channel:** {v['channel_title']}")
                st.write(f"**Published:** {v['published_at'][:10]} ({days_old} days ago)")
                st.write(f"**Duration:** {iso8601_to_hms(v['duration_iso'])} ({duration_seconds}s)")
                if v.get("tags"):
                    st.write(f"**Tags:** {', '.join(v['tags'][:10])}")

            # ===== CORE METRICS =====
            st.markdown("### 📊 Core Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("👁️ Views", f"{v['views']:,}")
            m2.metric("👍 Likes", f"{v['likes']:,}")
            m3.metric("💬 Comments", f"{v['comments_count']:,}")
            m4.metric("📈 Engagement", f"{engagement_rate:.4f}")

            # ===== AD DURATION ANALYSIS =====
            st.markdown("### ⏱️ Ad Duration Optimization")
            duration_msg, duration_type = analyze_video_duration_for_ads(duration_seconds)
            if duration_type == "success":
                st.success(duration_msg)
            elif duration_type == "warning":
                st.warning(duration_msg)
            else:
                st.error(duration_msg)

            # ===== TITLE OPTIMIZATION =====
            st.markdown("### 📝 Title Optimization Score")
            title_score, title_recommendations = analyze_title_for_ads(v["title"])
            
            col_score, col_bar = st.columns([1, 3])
            with col_score:
                st.metric("Score", f"{title_score}/100")
            with col_bar:
                st.progress(title_score / 100)
            
            for rec in title_recommendations:
                st.write(rec)

            # ===== ENGAGEMENT VELOCITY =====
            st.markdown("### 🚀 Engagement Velocity")
            st.caption("Performance metrics normalized by days since publication")
            velocity = analyze_engagement_velocity(v["views"], v["likes"], v["comments_count"], days_old)
            
            vel1, vel2, vel3, vel4 = st.columns(4)
            vel1.metric("Views/Day", f"{velocity['views_per_day']:,.0f}")
            vel2.metric("Likes/Day", f"{velocity['likes_per_day']:,.1f}")
            vel3.metric("Comments/Day", f"{velocity['comments_per_day']:,.1f}")
            vel4.metric("Engagement/Day", f"{velocity['engagement_per_day']:,.1f}")

            # ===== CTA ANALYSIS =====
            st.markdown("### 🎯 Call-to-Action Analysis")
            cta_count = detect_cta_in_description(v.get("description", ""))
            if cta_count > 0:
                st.success(f"✅ Found {cta_count} CTA elements in description")
            else:
                st.warning("⚠️ No clear CTAs detected - consider adding links or action prompts")
            
            with st.expander("📄 View Full Description"):
                st.text(v.get("description", "No description available")[:1000] + "..." if len(v.get("description", "")) > 1000 else v.get("description", ""))

            # ===== CHANNEL BENCHMARK =====
            use_channel = channel_id if channel_id else v.get("channel_id")
            if use_channel:
                st.markdown("### 📈 Channel Performance Benchmark")
                
                with st.spinner("📊 Analyzing channel average..."):
                    avg_eng_rate = get_channel_engagement(api_key, use_channel)
                    
                    if not np.isnan(avg_eng_rate):
                        viral_score, viral_msg, viral_status = calculate_viral_coefficient(
                            v["views"], v["likes"], v["comments_count"], avg_eng_rate
                        )
                        
                        col_vid, col_chan, col_viral = st.columns(3)
                        col_vid.metric("Video Engagement", f"{engagement_rate:.4f}")
                        col_chan.metric("Channel Average", f"{avg_eng_rate:.4f}")
                        if viral_score:
                            col_viral.metric("Viral Score", f"{viral_score:.0f}%")
                        
                        if viral_status == "success":
                            st.success(viral_msg)
                        elif viral_status == "warning":
                            st.warning(viral_msg)
                        else:
                            st.info(viral_msg)
                    else:
                        st.info("⚠️ Could not calculate channel average")
            else:
                st.info("💡 Provide Channel ID for performance comparison")

            # ===== TOP KEYWORDS =====
            st.markdown("### 🔍 Top Keywords Extracted")
            keywords = extract_top_keywords(v["title"], v.get("tags", []), v.get("description", ""))
            if keywords:
                st.write(", ".join([f"`{k}`" for k in keywords[:10]]))
            else:
                st.info("No keywords extracted")

            # ===== THUMBNAIL ANALYSIS =====
            st.markdown("### 🖼️ Advanced Thumbnail Analysis")
            st.caption("AI-powered analysis of brightness, contrast, dominant colors, text (OCR), and faces")
            thumb_analysis = analyze_thumbnail(v["thumbnail"]) if v.get("thumbnail") else {"ok": False, "error": "No thumbnail"}
            
            if not thumb_analysis["ok"]:
                st.warning(f"⚠️ Thumbnail analysis unavailable: {thumb_analysis.get('error')}")
            else:
                col_t1, col_t2 = st.columns([2, 1])
                
                with col_t1:
                    st.write(f"**Size:** {thumb_analysis.get('size')}")
                    st.write(f"**Brightness (0-255):** {thumb_analysis.get('brightness'):.1f}")
                    st.write(f"**Contrast (stddev):** {thumb_analysis.get('contrast'):.1f}")
                    
                    brightness = thumb_analysis.get('brightness', 0)
                    if brightness < 80:
                        st.write("💡 **Recommendation:** Thumbnail is dark - consider brightening for better visibility")
                    elif brightness > 200:
                        st.write("💡 **Recommendation:** Thumbnail is very bright - may need more contrast")
                    else:
                        st.write("✅ **Brightness:** Optimal range for visibility")
                    
                    ocr_val = thumb_analysis.get("ocr_text")
                    if ocr_val is None:
                        st.write("**Text detected:** None")
                    elif isinstance(ocr_val, str) and ocr_val.startswith("OCR error:"):
                        st.write(f"**OCR:** {ocr_val}")
                    else:
                        st.write(f"**Text detected:** {ocr_val[:200]}")
                    
                    fc = thumb_analysis.get("face_count")
                    if fc is None:
                        st.write("**Faces detected:** OpenCV not available")
                    elif isinstance(fc, int):
                        st.write(f"**Faces detected:** {fc}")
                        if fc > 0:
                            st.write("✅ **Recommendation:** Human faces increase engagement!")
                    else:
                        st.write(f"**Faces:** {fc}")
                
                with col_t2:
                    rgb = thumb_analysis.get('dominant_color_rgb')
                    hexc = thumb_analysis.get('dominant_color_hex')
                    if rgb and hexc:
                        st.markdown("**Dominant Color**")
                        st.markdown(f"RGB: {rgb}")
                        st.markdown(f"HEX: {hexc}")
                        st.markdown(
                            f'<div style="width:100%;height:80px;border-radius:8px;border:2px solid #ddd;background:{hexc};margin-top:10px"></div>',
                            unsafe_allow_html=True
                        )

            # ===== VIDEO PREVIEW THUMBNAILS =====
            st.markdown("### 🎬 Video Preview Moments")
            st.caption("YouTube auto-generated thumbnails from different video segments")
            thumbs = get_video_preview_thumbnails(vid)
            col_thumb1, col_thumb2, col_thumb3 = st.columns(3)
            with col_thumb1:
                st.image(thumbs["beginning"], caption="Beginning", width=280)
            with col_thumb2:
                st.image(thumbs["middle"], caption="Middle", width=280)
            with col_thumb3:
                st.image(thumbs["end"], caption="End", width=280)

            # ===== COMMENT ANALYSIS =====
            comments_df = fetch_comments(api_key, vid, max_comments) if max_comments > 0 else pd.DataFrame()

            if max_comments > 0 and not comments_df.empty:
                st.markdown("### 💭 Audience Sentiment Analysis")
                sentiment_data = analyze_comment_sentiment_detailed(comments_df)
                
                if sentiment_data:
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("😊 Positive", f"{sentiment_data['positive_pct']:.1f}%", 
                                 delta=f"{sentiment_data['positive']} comments")
                    col_s2.metric("😐 Neutral", f"{sentiment_data['neutral_pct']:.1f}%",
                                 delta=f"{sentiment_data['neutral']} comments")
                    col_s3.metric("😞 Negative", f"{sentiment_data['negative_pct']:.1f}%",
                                 delta=f"{sentiment_data['negative']} comments")
                    col_s4.metric("📊 Avg Score", f"{sentiment_data['avg_sentiment']:.3f}")
                    
                    if sentiment_data['positive_pct'] > 60:
                        st.success("✅ Overwhelmingly positive audience response - excellent for brand reputation")
                    elif sentiment_data['negative_pct'] > 30:
                        st.warning("⚠️ Significant negative sentiment detected - review feedback for improvements")
                    else:
                        st.info("📊 Mixed sentiment - typical for advertising content")

                st.markdown("### 🗣️ Comment Word Cloud")
                all_comments = " ".join(comments_df["comment_text"].values)
                st.image(generate_wordcloud(all_comments))

                st.markdown("### 📊 Most Frequent Words in Comments")
                plot_top_comment_words(comments_df, n=12)

                st.markdown("### 🧭 Emotion Map & Topic Detection")
                st.caption("Advanced AI analysis of audience emotions and discussion topics")
                emo = emotion_and_topics_from_comments(comments_df, n_topics=3)
                
                if emo.get("sentiment_counts"):
                    st.write(f"**Sentiment Distribution:** {emo.get('sentiment_counts')}")
                    st.write(f"**Average Sentiment Score:** {emo.get('avg_sentiment'):.3f}")
                    
                    st.markdown("**Representative Comments by Sentiment**")
                    for bucket, comments_list in emo.get("repr_comments", {}).items():
                        if comments_list:
                            with st.expander(f"💬 {bucket.title()} Comments ({len(comments_list)})"):
                                for i, comment in enumerate(comments_list[:3], 1):
                                    st.write(f"{i}. {comment}")
                    
                    if emo.get("topics"):
                        st.markdown("**Detected Discussion Topics (LDA)**")
                        for topic in emo["topics"]:
                            st.write(f"**Topic {topic['topic_id'] + 1}:** {', '.join(topic['top_words'][:6])}")
                    else:
                        st.info("💡 Not enough data for topic modeling")

                with st.expander("📝 View Sample Comments"):
                    st.dataframe(comments_df.head(20)[["comment_text", "like_count"]])

            elif max_comments > 0:
                st.info("ℹ️ No comments available for this video")

            # ===== CHANNEL TAG HISTORY & LENGTH ANALYSIS =====
            if use_channel:
                st.markdown("### 📚 Channel Deep Dive Analysis")
                st.caption("Historical tag usage and video length vs engagement patterns")
                
                with st.spinner("🔍 Analyzing channel history..."):
                    tags_counter, ch_df = channel_tags_history(api_key, use_channel, max_videos=50)
                
                if tags_counter:
                    col_tags1, col_tags2 = st.columns([1, 1])
                    
                    with col_tags1:
                        st.markdown("**Top Tags Used by Channel**")
                        top_tags = tags_counter.most_common(15)
                        df_tags = pd.DataFrame(top_tags, columns=["tag", "count"])
                        st.dataframe(df_tags)
                    
                    with col_tags2:
                        st.markdown("**Tag Frequency Distribution**")
                        st.bar_chart(df_tags.set_index("tag")["count"])
                else:
                    st.info("No tag data available")

                if not ch_df.empty:
                    ch_df["engagement_rate"] = (ch_df["likes"] + ch_df["comments_count"]) / ch_df["views"].replace(0, np.nan)
                    ch_df["duration_s"] = ch_df["duration_iso"].apply(iso8601_to_seconds)
                    
                    st.markdown("**Video Duration vs Engagement Rate**")
                    st.caption("Scatter plot showing relationship between video length and audience engagement")
                    
                    scatter_df = ch_df.dropna(subset=["duration_s", "engagement_rate"])
                    if not scatter_df.empty:
                        chart = alt.Chart(scatter_df).mark_circle(size=80, opacity=0.6).encode(
                            x=alt.X('duration_s:Q', title='Duration (seconds)', scale=alt.Scale(zero=False)),
                            y=alt.Y('engagement_rate:Q', title='Engagement Rate'),
                            color=alt.Color('engagement_rate:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                            tooltip=['title:N', 'duration_s:Q', 'engagement_rate:Q', 'views:Q']
                        ).properties(
                            height=400
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
                        
                        avg_dur = scatter_df["duration_s"].mean()
                        avg_eng = scatter_df["engagement_rate"].mean()
                        st.write(f"📊 **Channel Insights:** Average duration: {avg_dur:.0f}s ({avg_dur/60:.1f}min) | Average engagement: {avg_eng:.4f}")
                    else:
                        st.info("Not enough data for scatter plot")

                st.markdown("### ⏳ Engagement Timeline")
                plot_channel_history(api_key, use_channel, n=15)

            # ===== EXPORT DATA =====
            st.markdown("---")
            st.markdown("### 📥 Export Analysis")
            df_out = pd.DataFrame([{
                **v,
                "engagement_rate": engagement_rate,
                "duration_seconds": duration_seconds,
                "days_since_publish": days_old,
                "views_per_day": velocity['views_per_day'],
                "title_score": title_score,
                "cta_count": cta_count
            }])
            st.download_button(
                "📥 Download Full Analysis (CSV)",
                df_out.to_csv(index=False).encode(),
                "youtube_ad_analysis.csv",
                "text/csv"
            )

    # ===== SAVE ANALYSIS =====
    if st.session_state["last_analysis"] is not None and st.session_state.get("analysis_done"):
        if st.button("💾 Save Analysis to Dashboard"):
            st.session_state["portfolio"].append(st.session_state["last_analysis"])
            try:
                append_to_sheet(st.session_state["last_analysis"])
                st.success("✅ Analysis saved to Dashboard and Google Sheets!")
            except Exception as e:
                st.warning(f"⚠️ Saved locally but Google Sheets sync failed: {e}")

    # ========== VIDEO COMPARISON ==========
    if compare:
        st.session_state["comparison_done"] = True
        st.session_state["analysis_done"] = False
        

        vid1 = extract_video_id(video_url)
        vid2 = extract_video_id(video_url_2)

        if not vid1 or not vid2:
            st.error("❌ Invalid URL for one or both videos.")
            st.stop()

        with st.spinner("🔍 Comparing videos..."):
            v1 = get_video_details(api_key, vid1)
            v2 = get_video_details(api_key, vid2)
            if not v1 or not v2:
                st.error("❌ Could not retrieve video information.")
                st.stop()

            eng1 = (v1["likes"] + v1["comments_count"]) / max(v1["views"], 1)
            eng2 = (v2["likes"] + v2["comments_count"]) / max(v2["views"], 1)
            
            dur1 = iso8601_to_seconds(v1["duration_iso"])
            dur2 = iso8601_to_seconds(v2["duration_iso"])
            
            days1 = days_since(v1["published_at"])
            days2 = days_since(v2["published_at"])
            
            vel1 = analyze_engagement_velocity(v1["views"], v1["likes"], v1["comments_count"], days1)
            vel2 = analyze_engagement_velocity(v2["views"], v2["likes"], v2["comments_count"], days2)

            st.markdown("### 🆚 Head-to-Head Video Comparison")
            
            colc1, colc2 = st.columns(2)
            
            with colc1:
                st.subheader("📹 Video 1")
                if v1.get("thumbnail"):
                    st.image(v1["thumbnail"], width=380)
                st.write(f"**{v1['title']}**")
                st.write(f"Channel: {v1['channel_title']}")
                st.write(f"Duration: {iso8601_to_hms(v1['duration_iso'])}")
                st.write(f"Published: {days1} days ago")
                
                st.metric("Views", f"{v1['views']:,}")
                st.metric("Likes", f"{v1['likes']:,}")
                st.metric("Comments", f"{v1['comments_count']:,}")
                st.metric("Engagement Rate", f"{eng1:.4f}")
                st.metric("Views/Day", f"{vel1['views_per_day']:,.0f}")
                
            with colc2:
                st.subheader("📹 Video 2")
                if v2.get("thumbnail"):
                    st.image(v2["thumbnail"], width=380)
                st.write(f"**{v2['title']}**")
                st.write(f"Channel: {v2['channel_title']}")
                st.write(f"Duration: {iso8601_to_hms(v2['duration_iso'])}")
                st.write(f"Published: {days2} days ago")
                
                st.metric("Views", f"{v2['views']:,}", 
                         delta=f"{((v2['views']/v1['views']-1)*100):+.1f}%" if v1['views'] > 0 else None)
                st.metric("Likes", f"{v2['likes']:,}", 
                         delta=f"{((v2['likes']/v1['likes']-1)*100):+.1f}%" if v1['likes'] > 0 else None)
                st.metric("Comments", f"{v2['comments_count']:,}", 
                         delta=f"{((v2['comments_count']/v1['comments_count']-1)*100):+.1f}%" if v1['comments_count'] > 0 else None)
                st.metric("Engagement Rate", f"{eng2:.4f}", 
                         delta=f"{((eng2/eng1-1)*100):+.1f}%" if eng1 > 0 else None)
                st.metric("Views/Day", f"{vel2['views_per_day']:,.0f}", 
                         delta=f"{((vel2['views_per_day']/vel1['views_per_day']-1)*100):+.1f}%" if vel1['views_per_day'] > 0 else None)

            st.markdown("### 🏆 Performance Winner")
            
            points_v1 = 0
            points_v2 = 0
            
            comparison_results = []
            
            if eng1 > eng2:
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has higher engagement rate")
            elif eng2 > eng1:
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has higher engagement rate")
            else:
                comparison_results.append("➡️ Equal engagement rate")
            
            if vel1['views_per_day'] > vel2['views_per_day']:
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has better view velocity")
            elif vel2['views_per_day'] > vel1['views_per_day']:
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has better view velocity")
            
            if dur1 <= 30 and dur2 > 30:
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has better ad-optimized duration")
            elif dur2 <= 30 and dur1 > 30:
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has better ad-optimized duration")
            
            if v1['likes']/max(v1['views'], 1) > v2['likes']/max(v2['views'], 1):
                points_v1 += 1
                comparison_results.append("✅ **Video 1** has better like ratio")
            elif v2['likes']/max(v2['views'], 1) > v1['likes']/max(v1['views'], 1):
                points_v2 += 1
                comparison_results.append("✅ **Video 2** has better like ratio")
            
            for result in comparison_results:
                st.write(result)
            
            st.markdown("---")
            if points_v1 > points_v2:
                st.success(f"🏆 **Video 1 WINS** ({points_v1} vs {points_v2} points)")
            elif points_v2 > points_v1:
                st.success(f"🏆 **Video 2 WINS** ({points_v2} vs {points_v1} points)")
            else:
                st.info(f"🤝 **TIE** ({points_v1} vs {points_v2} points)")
            
            st.markdown("### 📥 Export Comparison")
            comparison_df = pd.DataFrame([
                {
                    "video": "Video 1", 
                    "title": v1["title"], 
                    "views": v1["views"], 
                    "likes": v1["likes"],
                    "comments": v1["comments_count"],
                    "engagement": eng1, 
                    "views_per_day": vel1['views_per_day'],
                    "duration_s": dur1
                },
                {
                    "video": "Video 2", 
                    "title": v2["title"], 
                    "views": v2["views"], 
                    "likes": v2["likes"],
                    "comments": v2["comments_count"],
                    "engagement": eng2, 
                    "views_per_day": vel2['views_per_day'],
                    "duration_s": dur2
                }
            ])
            
            st.download_button(
                "📥 Download Comparison (CSV)",
                comparison_df.to_csv(index=False).encode(),
                "video_comparison.csv",
                "text/csv"
            )

# =========================================================
# TAB 2: DASHBOARD
# =========================================================
with tab2:
    st.header("📁 Saved Analyses Portfolio")
    st.caption("Track and compare all your analyzed videos over time")

    if st.session_state["portfolio"]:
        dfp = pd.DataFrame(st.session_state["portfolio"])
        
        if "engagement_rate" in dfp.columns:
            dfp["engagement_rate"] = pd.to_numeric(dfp["engagement_rate"], errors="coerce")

        st.markdown("### 📊 Portfolio Summary")
        colA, colB, colC = st.columns(3)

        colA.metric("Total Analyses", len(dfp))

        if "engagement_rate" in dfp.columns and dfp["engagement_rate"].notna().any():
            best_eng = dfp["engagement_rate"].max()
            avg_eng = dfp["engagement_rate"].mean()
            colB.metric("Best Engagement", f"{best_eng:.5f}", delta=f"Avg: {avg_eng:.5f}")
        else:
            colB.metric("Best Engagement", "—")

        if "date" in dfp.columns and dfp["date"].notna().any():
            colC.metric("Last Saved", dfp["date"].iloc[-1])
        else:
            colC.metric("Last Saved", "—")

        st.markdown("### 📋 All Saved Analyses")
        st.dataframe(dfp, use_container_width=True)

        if (
            "date" in dfp.columns
            and "engagement_rate" in dfp.columns
            and dfp["engagement_rate"].notna().any()
        ):
            st.markdown("### 📈 Engagement Timeline")
            st.caption("Track engagement rate trends across all analyzed videos")
            st.line_chart(dfp.set_index("date")["engagement_rate"])

        if "title" in dfp.columns and "engagement_rate" in dfp.columns:
            st.markdown("### 🏅 Top Performing Videos")
            top_df = (
                dfp.dropna(subset=["engagement_rate"])
                .sort_values("engagement_rate", ascending=False)
                .head(5)
            )
            if len(top_df):
                st.bar_chart(top_df.set_index("title")["engagement_rate"])
            else:
                st.info("No engagement data yet. Save an analysis first.")
        
        st.markdown("### 📥 Export Portfolio")
        st.download_button(
            "📥 Download Full Portfolio (CSV)",
            dfp.to_csv(index=False).encode(),
            "portfolio_full.csv",
            "text/csv"
        )
    else:
        st.info("📭 No saved analyses yet. Go to 'Video Analysis', run an analysis, and click 'Save Analysis to Dashboard'.")

