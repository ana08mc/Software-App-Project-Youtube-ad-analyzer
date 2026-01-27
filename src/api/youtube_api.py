import time
import numpy as np
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build

from collections import Counter

from utils.helpers import clean_html

# =========================================================
# YOUTUBE API FUNCTIONS
# =========================================================
def get_video_details(api_key, vid):
    """Fetch video details from YouTube API."""
    yt = build("youtube", "v3", developerKey=api_key)
    r = yt.videos().list(part="snippet,statistics,contentDetails", id=vid).execute()
    if not r.get("items"):
        return None
    it = r["items"][0]
    sn, stt, cd = it.get("snippet", {}), it.get("statistics", {}), it.get("contentDetails", {})
    return {
        "video_id": it["id"],
        "title": sn.get("title", ""),
        "channel_title": sn.get("channelTitle", ""),
        "channel_id": sn.get("channelId", ""),
        "published_at": sn.get("publishedAt", ""),
        "description": sn.get("description", ""),
        "tags": sn.get("tags", []),
        "thumbnail": sn.get("thumbnails", {}).get("high", {}).get("url"),
        "views": int(stt.get("viewCount", 0)),
        "likes": int(stt.get("likeCount", 0)),
        "comments_count": int(stt.get("commentCount", 0)),
        "duration_iso": cd.get("duration", ""),
    }

def fetch_comments(api_key, vid, cap):
    """Fetch comments from a video."""
    if cap == 0:
        return pd.DataFrame(columns=["comment_text", "like_count", "published_at"])
    yt = build("youtube", "v3", developerKey=api_key)
    out, got, page = [], 0, None
    while got < cap:
        try:
            r = yt.commentThreads().list(
                part="snippet",
                videoId=vid,
                maxResults=min(100, cap - got),
                pageToken=page,
                order="relevance",
                textFormat="html"
            ).execute()
        except Exception:
            break
        for it in r.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "comment_text": clean_html(top.get("textDisplay", "")),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt", "")
            })
        got += len(r.get("items", []))
        page = r.get("nextPageToken")
        if not page:
            break
        time.sleep(0.1)
    return pd.DataFrame(out)

def get_channel_videos(api_key, channel_id, max_results=30):
    """Get list of video IDs from a channel."""
    yt = build("youtube", "v3", developerKey=api_key)
    videos = []
    next_page = None
    while len(videos) < max_results:
        request = yt.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page,
            type="video"
        )
        response = request.execute()
        for item in response.get("items", []):
            videos.append(item["id"]["videoId"])
        next_page = response.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.3)
    return videos

def get_channel_engagement(api_key, channel_id, max_results=30):
    """Calculate average engagement rate for a channel."""
    videos = get_channel_videos(api_key, channel_id, max_results)
    rates = []
    for vid in videos:
        v = get_video_details(api_key, vid)
        if v:
            rate = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
            rates.append(rate)
    return np.mean(rates) if rates else np.nan



# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================

def plot_channel_history(api_key, channel_id, n=15):
    """Plot channel engagement history."""
    videos = get_channel_videos(api_key, channel_id, max_results=n)
    data = []
    for vid in videos:
        v = get_video_details(api_key, vid)
        if v:
            v["engagement_rate"] = (v["likes"] + v["comments_count"]) / max(v["views"], 1)
            data.append(v)
    if data:
        df = pd.DataFrame(data).sort_values("published_at")
        st.line_chart(df.set_index("published_at")["engagement_rate"])
    else:
        st.info("No data available for channel history.")
