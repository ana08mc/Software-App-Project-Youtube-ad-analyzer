# YouTube Ad Analyzer – Marketing Insights Pro

**M2 Software Project 2025–2026 – Aix-Marseille Université**

This Streamlit web application analyzes YouTube videos from a **marketing and advertising perspective**, combining:

- YouTube Data API
- Natural Language Processing (sentiment analysis, topic modeling)
- Computer Vision (OCR, face detection, color & brightness analysis)
- Google Sheets as persistent storage
- A professional modular Python architecture
- Dockerized deployment

---

## Team

- **Ana Maria MARTINEZ CASTRO**
- **Vanessa MARTINEZ HERRERA**

---

## What the Application Does

This tool allows you to:

- Analyze video performance (views, likes, engagement rate)
- Evaluate advertising optimization (duration, title, CTA presence)
- Analyze audience sentiment and discussion topics (VADER + LDA)
- Analyze thumbnails using OCR, face detection, and color metrics
- Compare two videos side by side
- Automatically store analysis results in Google Sheets
- Visualize historical performance of a channel

---

## Project Architecture

```
src/
 ├── api/
 │    └── youtube_api.py        # Communication with YouTube API
 ├── analysis/
 │    ├── video_analysis.py     # Marketing logic, NLP, analytics
 │    ├── viz.py                # Visualizations
 │    └── thumbnail.py          # OCR and image analysis
 ├── storage/
 │    └── sheets.py             # Google Sheets integration
 ├── utils/
 │    └── helpers.py            # Utility helper functions
 └── app/
      └── streamlit_app.py     # Streamlit UI (interface only)
```

This separation ensures:

- Clear separation between UI, API logic, analysis, and storage
- Easier testing and maintenance
- Professional Python project organization

---

## Secrets Configuration

The app requires **two separate credentials**:

| Service | Variable | Purpose |
|---|---|---|
| YouTube Data API | `YT_API_KEY` | Retrieve video and channel data |
| Google Sheets | `GOOGLE_CREDENTIALS_JSON` | Store and load analysis results |

The repository includes a `.env.example` file.

To run the project, copy it:

```bash
cp .env.example .env
```

and fill in your own credentials.

---

### A) YouTube API Key

1. Go to Google Cloud Console
2. Enable **YouTube Data API v3**
3. Create Credentials → API Key
4. Add to `.env`:

```
YT_API_KEY=YOUR_YOUTUBE_API_KEY
```

---

### B) Google Sheets Service Account

1. Enable **Google Sheets API** and **Google Drive API**
2. Create a Service Account
3. Download the JSON credentials file
4. Share your Google Sheet with the service account email
5. Paste the entire JSON (in one line) into `.env`:

```
GOOGLE_CREDENTIALS_JSON=PASTE_JSON_HERE
```

⚠️ Never upload this JSON file to GitHub.

---

## Run Locally

From the project root:

```bash
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

The app will be available at:  
http://localhost:8501

---

## Run with Docker (Recommended)

1. Create your `.env` file from the template:

```bash
cp .env.example .env
```

2. Fill in your credentials

3. Start the application:

```bash
docker compose up --build
```

4. Open in your browser:

http://localhost:8501

---

## Dependencies

All required Python packages are listed in `requirements.txt`.

---

## Academic Information

**Course:** M2 Software Project 2025-2026  
**University:** Aix-Marseille Université  
**Professor:** Virgile Pesce

---

## License

MIT License
