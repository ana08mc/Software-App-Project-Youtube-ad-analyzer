import os
import io
import requests
import numpy as np
from collections import Counter

# Optional libraries
try:
    from PIL import Image, ImageStat
except Exception:
    Image = None
    ImageStat = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# =========================================================
# THUMBNAIL ANALYZER (ADVANCED)
# =========================================================
def _rgb_to_hex(rgb):
    """Convert RGB tuple to hex color."""
    try:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    except Exception:
        return None

def analyze_thumbnail(url):
    """Comprehensive thumbnail analysis."""
    out = {
        "ok": False,
        "error": None,
        "brightness": None,
        "contrast": None,
        "dominant_color_rgb": None,
        "dominant_color_hex": None,
        "ocr_text": None,
        "face_count": None,
        "size": None,
    }
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        if Image is None:
            out["error"] = "Pillow (PIL) not installed"
            return out
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        out["size"] = img.size

        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]
        contrast = stat.stddev[0]
        out["brightness"] = float(brightness)
        out["contrast"] = float(contrast)

        arr = np.array(img).reshape(-1, 3)
        arr_bucket = (arr // 32) * 32
        tuples = [tuple(int(x) for x in row) for row in arr_bucket]
        if tuples:
            dom_rgb = Counter(tuples).most_common(1)[0][0]
            out["dominant_color_rgb"] = dom_rgb
            out["dominant_color_hex"] = _rgb_to_hex(dom_rgb)

        if pytesseract is not None:
            try:
                t_cmd = os.getenv("TESSERACT_CMD")
                if t_cmd:
                    pytesseract.pytesseract.tesseract_cmd = t_cmd
                ocr_text = pytesseract.image_to_string(img)
                out["ocr_text"] = ocr_text.strip() if ocr_text and ocr_text.strip() else None
            except Exception as e:
                out["ocr_text"] = f"OCR error: {e}"

        if cv2 is not None:
            try:
                np_img = np.array(img)[:, :, ::-1].copy()
                gray_cv = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
                cascade_path = None
                if hasattr(cv2.data, "haarcascades"):
                    candidate = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                    if os.path.exists(candidate):
                        cascade_path = candidate
                if cascade_path:
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    faces = face_cascade.detectMultiScale(gray_cv, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    out["face_count"] = int(len(faces))
                else:
                    out["face_count"] = None
            except Exception as e:
                out["face_count"] = f"Face detect error: {e}"

        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

