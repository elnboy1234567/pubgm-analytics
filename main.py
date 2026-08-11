import re
import cv2
import pytesseract
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
app = FastAPI(title="PUBGM Analytics")
# ============================================================
# PUBG MOBILE RESULT TABLE
# Based on the real coordinates obtained from OCR diagnostics.
# ============================================================
TABLE = (0.21, 0.47, 0.80, 0.695)
# Real row centres discovered from the supplied screenshot.
# These are approximate positions inside the table crop.
ROW_CENTERS = [
    10,
    90,
    172,
    254,
]
# Horizontal positions inside the table crop.
COLUMNS = {
    "player": 75,
    "kills": 665,
    "assists": 797,
    "damage": 900,
    "survival": 994,
    "hp_recovered": 1140,
    "rescues": 1270,
    "return": 1388,
    "score": 1486,
}
# ============================================================
# BASIC OCR HELPERS
# ============================================================
def normalize_text(text):
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text
def clean_player_name(text):
    text = normalize_text(text)
    # Remove obvious OCR punctuation.
    text = re.sub(
        r"^[|Il!]+",
        "",
        text
    )
    text = re.sub(
        r"[|]+$",
        "",
        text
    )
    text = text.strip()
    # PUBG names normally contain letters/numbers,
    # sometimes underscores, hyphens, dots or asterisks.
    cleaned = re.sub(
        r"[^A-Za-z0-9_*.\-]",
        "",
        text
    )
    return cleaned
def normalize_number(text):
    if not text:
        return None
    text = str(text).strip()
    # Common OCR substitutions.
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "B": "8",
        "G": "6",
        "q": "9",
        "J": "1",
        "£": "2",
    }
    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )
    text = text.replace(
        ",",
        "."
    )
    # Keep numbers and decimal point.
    text = re.sub(
        r"[^0-9.]",
        "",
        text
    )
    if not text:
        return None
    return text
def parse_integer(text):
    text = normalize_number(text)
    if not text:
        return None
    match = re.search(
        r"\d+",
        text
    )
    if not match:
        return None
    try:
        return int(
            match.group()
        )
    except Exception:
        return None
def parse_score(text):
    text = normalize_number(text)
    if not text:
        return None
    match = re.search(
        r"\d+(?:\.\d+)?",
        text
    )
    if not match:
        return None
    try:
        value = float(
            match.group()
        )
        if 0 <= value <= 150:
            return value
    except Exception:
        pass
    return None
def parse_survival(text):
    if not text:
        return None
    text = str(text)
    # Typical PUBG survival values:
    # 0:00 - 60:00 approximately.
    numbers = re.findall(
        r"\d+(?:[.:]\d+)?",
        text
    )
    if not numbers:
        return None
    # Look for something resembling minutes.
    for value in numbers:
        value = value.replace(
            ":",
            "."
        )
        try:
            number = float(
                value
            )
            if 0 < number <= 60:
                return number
        except Exception:
            continue
    return None
# ============================================================
# OCR
# ============================================================
def perform_ocr(table):
    gray = cv2.cvtColor(
        table,
        cv2.COLOR_BGR2GRAY
    )
    # One resize only.
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    data = pytesseract.image_to_data(
        gray,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )
    words = []
    total = len(
        data["text"]
    )
    for i in range(total):
        text = normalize_text(
            data["text"][i]
        )
        if not text:
            continue
        try:
            confidence = float(
                data["conf"][i]
            )
        except Exception:
            confidence = -1
        if confidence < 15:
            continue
        x = int(
            data["left"][i]
        ) / 2
        y = int(
            data["top"][i]
        ) / 2
        width = int(
            data["width"][i]
        ) / 2
        height = int(
            data["height"][i]
        ) / 2
        words.append(
            {
                "text": text,
                "confidence": confidence,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "center_x": x + width / 2,
                "center_y": y + height / 2,
            }
        )
    return words
# ============================================================
# ROW DETECTION
# ============================================================
def detect_rows(words, table_height):
    numeric_words = []
    for word in words:
        text = word["text"]
        # Ignore border-like characters.
        if text in [
            "|",
            "_",
            "-",
            "—",
        ]:
            continue
        numeric_words.append(
            word
        )
    if not numeric_words:
        return []
    # Cluster OCR words by vertical position.
    rows = []
    for word in sorted(
        numeric_words,
        key=lambda w: w["center_y"]
    ):
        cy = word["center_y"]
        selected = None
        for row in rows:
            if abs(
                cy - row["center_y"]
            ) <= 25:
                selected = row
                break
        if selected is None:
            selected = {
                "center_y": cy,
                "words": []
            }
            rows.append(
                selected
            )
        selected["words"].append(
            word
        )
        ys = [
            item["center_y"]
            for item in selected["words"]
        ]
        selected["center_y"] = (
            sum(ys) / len(ys)
        )
    # Sort vertically.
    rows.sort(
        key=lambda row:
        row["center_y"]
    )
    # PUBG results normally have max 4 players.
    if len(rows) > 4:
        # Keep the four strongest rows by amount
        # of useful OCR content.
        rows = sorted(
            rows,
            key=lambda row:
            len(row["words"]),
            reverse=True
        )[:4]
        rows.sort(
            key=lambda row:
            row["center_y"]
        )
    return rows
# ============================================================
# FIND WORD CLOSEST TO A COLUMN
# ============================================================
def nearest_word(
    words,
    x_target,
    max_distance=75
):
    candidates = []
    for word in words:
        distance = abs(
            word["center_x"] -
            x_target
        )
        if distance <= max_distance:
            candidates.append(
                (
                    distance,
                    word
                )
            )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item:
        item[0]
    )
    return candidates[0][1]
# ============================================================
# EXTRACT PLAYER FROM ROW
# ============================================================
def extract_player(row):
    words = row["words"]
    candidates = []
    for word in words:
        # Name is on the left side.
        if word["center_x"] > 400:
            continue
        text = clean_player_name(
            word["text"]
        )
        if len(text) < 3:
            continue
        # Ignore pure numbers.
        if re.fullmatch(
            r"\d+",
            text
        ):
            continue
        candidates.append(
            (
                word["confidence"],
                len(text),
                text
            )
        )
    if not candidates:
        return None
    # Prefer confidence, then length.
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]
        ),
        reverse=True
    )
    return candidates[0][2]
# ============================================================
# EXTRACT ROW
# ============================================================
def extract_row(row):
    words = row["words"]
    player = extract_player(
        row
    )
    if not player:
        return None
    result = {
        "player": player,
        "kills": None,
        "assists": None,
        "damage": None,
        "survival": None,
        "hp_recovered": None,
        "rescues": None,
        "return": None,
        "score": None,
    }
    # --------------------------------------------------------
    # KILLS
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["kills"],
        65
    )
    if word:
        result["kills"] = parse_integer(
            word["text"]
        )
    # --------------------------------------------------------
    # ASSISTS
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["assists"],
        65
    )
    if word:
        result["assists"] = parse_integer(
            word["text"]
        )
    # --------------------------------------------------------
    # DAMAGE
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["damage"],
        65
    )
    if word:
        result["damage"] = parse_integer(
            word["text"]
        )
    # --------------------------------------------------------
    # SURVIVAL
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["survival"],
        95
    )
    if word:
        result["survival"] = parse_survival(
            word["text"]
        )
    # --------------------------------------------------------
    # HP RECOVERED
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["hp_recovered"],
        65
    )
    if word:
        result["hp_recovered"] = parse_integer(
            word["text"]
        )
    # --------------------------------------------------------
    # RESCUES
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["rescues"],
        65
    )
    if word:
        result["rescues"] = parse_integer(
            word["text"]
        )
    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["return"],
        65
    )
    if word:
        result["return"] = parse_integer(
            word["text"]
        )
    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------
    word = nearest_word(
        words,
        COLUMNS["score"],
        70
    )
    if word:
        result["score"] = parse_score(
            word["text"]
        )
    return result
# ============================================================
# METADATA
# ============================================================
MAPS = [
    "Erangel",
    "Miramar",
    "Rondo",
    "Sanhok",
    "Livik",
    "Vikendi",
    "Karakin",
    "Nusa",
]
def parse_metadata(img):
    h, w = img.shape[:2]
    header = img[
        int(0.07 * h):
        int(0.47 * h),
        int(0.20 * w):
        int(0.82 * w)
    ]
    gray = cv2.cvtColor(
        header,
        cv2.COLOR_BGR2GRAY
    )
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )
    lower = text.lower()
    map_name = None
    for game_map in MAPS:
        if game_map.lower() in lower:
            map_name = game_map
            break
    return {
        "placement": None,
        "map": map_name
    }
# ============================================================
# ANALYZE IMAGE
# ============================================================
def analyze_image(
    data,
    match_number
):
    array = np.frombuffer(
        data,
        np.uint8
    )
    img = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )
    if img is None:
        raise ValueError(
            "No se pudo leer la imagen"
        )
    h, w = img.shape[:2]
    x0, y0, x1, y1 = TABLE
    table = img[
        int(y0 * h):
        int(y1 * h),
        int(x0 * w):
        int(x1 * w)
    ]
    words = perform_ocr(
        table
    )
    rows = detect_rows(
        words,
        table.shape[0]
    )
    players = []
    for row in rows:
        player = extract_row(
            row
        )
        if not player:
            continue
        # Avoid duplicate players.
        if any(
            existing["player"].lower()
            ==
            player["player"].lower()
            for existing in players
        ):
            continue
        player["match"] = match_number
        players.append(
            player
        )
    metadata = parse_metadata(
        img
    )
    return {
        "match": match_number,
        "map": metadata["map"],
        "placement": metadata["placement"],
        "players": players
    }
# ============================================================
# WEB
# ============================================================
@app.get(
    "/",
    response_class=HTMLResponse
)
def home():
    return open(
        "static/index.html",
        encoding="utf-8"
    ).read()
@app.post("/api/analyze")
async def analyze(
    files: List[UploadFile] = File(...)
):
    results = []
    for i, file in enumerate(
        files,
        1
    ):
        try:
            data = await file.read()
            results.append(
                analyze_image(
                    data,
                    i
                )
            )
        except Exception as error:
            results.append(
                {
                    "match": i,
                    "error": str(error),
                    "players": []
                }
            )
    return JSONResponse(
        {
            "matches": results
        }
    )
@app.post("/api/export")
async def export(
    payload: dict
):
    return {
        "ok": True
    }
