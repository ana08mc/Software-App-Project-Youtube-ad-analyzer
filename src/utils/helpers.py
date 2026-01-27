"""
Helper functions module for processing YouTube videos.

This module contains utility functions used throughout the application:
- Video ID extraction
- Time format conversions
- HTML cleaning
- Date calculations
"""

import re
from datetime import datetime, timezone
from dateutil import parser as dtparser
from bs4 import BeautifulSoup


def extract_video_id(url_or_id: str) -> str:
    """
    Extracts a YouTube video ID from a URL or returns the ID if it is already valid.
    
    Args:
        url_or_id: YouTube URL or video ID (string)
        
    Returns:
        Video ID (11 characters) or an empty string if invalid
        
    Examples:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    # If it is already a valid ID (11 characters), return it directly
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id
    
    # Try extracting from standard URL (youtube.com/watch?v=...)
    match = re.search(r"v=([\w-]{11})", url_or_id)
    if match:
        return match.group(1)
    
    # Try extracting from short URL (youtu.be/...)
    match = re.search(r"youtu\.be/([\w-]{11})", url_or_id)
    if match:
        return match.group(1)
    
    return ""


def iso8601_to_hms(iso_duration: str) -> str:
    """
    Converts ISO 8601 duration to readable HH:MM:SS format.
    
    ISO 8601 is the format YouTube uses for video durations.
    Example: "PT1H2M30S" means 1 hour, 2 minutes, 30 seconds.
    
    Args:
        iso_duration: Duration in ISO 8601 format (e.g., "PT1H2M30S")
        
    Returns:
        Duration in HH:MM:SS format (e.g., "01:02:30")
        
    Examples:
        >>> iso8601_to_hms("PT1H2M30S")
        '01:02:30'
        >>> iso8601_to_hms("PT5M")
        '00:05:00'
        >>> iso8601_to_hms("PT45S")
        '00:00:45'
    """
    # Look for hours (H), minutes (M), and seconds (S) in the string
    h = re.search(r"(\d+)H", iso_duration or "")
    m = re.search(r"(\d+)M", iso_duration or "")
    s = re.search(r"(\d+)S", iso_duration or "")
    
    # Extract values or use 0 if not present
    hours = int(h.group(1)) if h else 0
    minutes = int(m.group(1)) if m else 0
    seconds = int(s.group(1)) if s else 0
    
    # Format as HH:MM:SS
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def iso8601_to_seconds(iso_duration: str) -> int:
    """
    Converts ISO 8601 duration to total seconds.
    
    Args:
        iso_duration: Duration in ISO 8601 format
        
    Returns:
        Total seconds (int)
        
    Examples:
        >>> iso8601_to_seconds("PT1H2M30S")
        3750
        >>> iso8601_to_seconds("PT5M")
        300
        >>> iso8601_to_seconds("PT45S")
        45
    """
    # Look for time components
    h = re.search(r"(\d+)H", iso_duration or "")
    m = re.search(r"(\d+)M", iso_duration or "")
    s = re.search(r"(\d+)S", iso_duration or "")
    
    # Extract values
    hours = int(h.group(1)) if h else 0
    minutes = int(m.group(1)) if m else 0
    seconds = int(s.group(1)) if s else 0
    
    # Compute total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    
    return total_seconds


def clean_html(text: str) -> str:
    """
    Removes HTML tags from a text string.
    
    YouTube sometimes returns comments with HTML formatting.
    This function extracts only the plain text.
    
    Args:
        text: Text that may contain HTML
        
    Returns:
        Clean text without HTML tags
        
    Examples:
        >>> clean_html("<b>Hello</b> <i>world</i>")
        'Hello world'
        >>> clean_html("Normal text")
        'Normal text'
    """
    if not text:
        return ""
    
    # Use BeautifulSoup to parse and extract text
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ", strip=True)
    
    return cleaned_text


def days_since(iso_date: str) -> int:
    """
    Calculates how many days have passed since an ISO date.
    
    Args:
        iso_date: Date in ISO 8601 format (e.g., "2024-01-15T10:30:00Z")
        
    Returns:
        Number of days (minimum 1)
        
    Examples:
        >>> # If today is 2024-01-20
        >>> days_since("2024-01-15T10:30:00Z")
        5
    """
    try:
        # Parse ISO date into datetime object
        dt = dtparser.isoparse(iso_date).astimezone(timezone.utc)
        
        # Compute difference with current date
        now = datetime.now(timezone.utc)
        days = (now - dt).days
        
        # Return at least 1 (to avoid division by zero)
        return max(days, 1)
        
    except Exception as e:
        # If parsing fails, return 1
        print(f"Error parsing date {iso_date}: {e}")
        return 1
