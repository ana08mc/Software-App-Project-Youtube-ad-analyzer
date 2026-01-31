import os
import json
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

SHEET_NAME = os.getenv("SHEET_NAME", "Marketing dashboard data")
EXPECTED_COLS = ["video_id", "title", "channel", "engagement_rate", "date"]

def _sanitize_json_env(raw: str) -> str:
    """
    Makes GOOGLE_CREDENTIALS_JSON more tolerant to:
    - surrounding quotes (single/double)
    - Windows CRLF
    - private_key with escaped \\n instead of real newlines
    """
    if raw is None:
        return ""

    raw = raw.strip()

    # Remove surrounding quotes if present
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1].strip()

    # Normalize CRLF
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    return raw

def get_sheet():
    """
    Returns a gspread worksheet (sheet1).
    If creds are missing/invalid, raises an Exception which the callers catch gracefully.
    """
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
    raw_json = _sanitize_json_env(raw_json)

    if not raw_json:
        raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON")

    # Parse JSON
    info = json.loads(raw_json)

    # Fix private_key newlines if they were escaped
    if "private_key" in info and isinstance(info["private_key"], str):
        info["private_key"] = info["private_key"].replace("\\n", "\n")

    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
    client = gspread.authorize(creds)

    # Open workbook + first sheet
    return client.open(SHEET_NAME).sheet1

def load_portfolio_from_sheet():
    """
    Loads rows from Sheets if possible.
    If not possible, returns [] WITHOUT breaking the app.
    """
    try:
        sheet = get_sheet()
        values = sheet.get_all_values()

        if not values:
            return []

        header = values[0]

        if set(EXPECTED_COLS).issubset(set(header)):
            return sheet.get_all_records()

        # fallback: coerce into expected columns
        fixed = []
        for row in values[1:]:
            fixed_row = row + [""] * (len(EXPECTED_COLS) - len(row))
            fixed.append(dict(zip(EXPECTED_COLS, fixed_row[:len(EXPECTED_COLS)])))
        return fixed

    except Exception as e:
        # Important: don't scare the user if Sheets isn't configured
        # Show a small info instead of a big yellow warning
        st.info("Google Sheets not configured (optional). Dashboard will work locally.")
        return []

def append_to_sheet(row_dict: dict):
    """
    Appends a single row to Sheets if configured.
    If not configured, raises Exception to be caught by the caller.
    """
    sheet = get_sheet()
    cols = EXPECTED_COLS
    row = [row_dict.get(c, "") for c in cols]
    sheet.append_row(row)
