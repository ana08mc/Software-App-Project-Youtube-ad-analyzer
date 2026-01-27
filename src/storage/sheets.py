import os
import json
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

SHEET_NAME = "Marketing dashboard data"

EXPECTED_COLS = ["video_id", "title", "channel", "engagement_rate", "date"]


# =========================================================
# GOOGLE SHEETS INTEGRATION
# =========================================================
SHEET_NAME = "Marketing dashboard data"

def get_sheet():
    """Get Google Sheet using JSON stored in environment variable."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]

    raw_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not raw_json:
        st.error("GOOGLE_CREDENTIALS_JSON secret not found. Add it in your deployment settings.")
        raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON")

    info = json.loads(raw_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1
    return sheet

EXPECTED_COLS = ["video_id", "title", "channel", "engagement_rate", "date"]

def load_portfolio_from_sheet():
    try:
        sheet = get_sheet()
        values = sheet.get_all_values()

        if not values:
            return []

        header = values[0]

        if set(EXPECTED_COLS).issubset(set(header)):
            rows = sheet.get_all_records()
            return rows

        fixed = []
        for row in values:
            fixed_row = row + [""] * (len(EXPECTED_COLS) - len(row))
            fixed.append(dict(zip(EXPECTED_COLS, fixed_row[:len(EXPECTED_COLS)])))
        return fixed

    except Exception as e:
        st.warning(f" Could not load data from Google Sheets: {e}")
        return []

def append_to_sheet(row_dict: dict):
    try:
        sheet = get_sheet()
        cols = ["video_id", "title", "channel", "engagement_rate", "date"]
        row = [row_dict.get(c, "") for c in cols]
        sheet.append_row(row)
    except Exception as e:
        st.error(f"Could not save to Google Sheets: {e}")