import re
import cv2
import pytesseract
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List

app = FastAPI(title="PUBGM Analytics MVP")


# ============================================================
# PUBG MOBILE RESULTS TABLE
# ============================================================

# Table coordinates relative to the complete screenshot.
TABLE = (0.21, 0.47, 0.80, 0.695)

# Maximum 4 players per match.
# These are relative vertical positions inside the table.
#
# We intentionally use fixed row positions instead of trying
# to detect rows with OCR. This prevents player names from
# being mixed together.
ROW_Y = [
    0.26,
    0.445,
    0.625,
    0.81,
]


# Horizontal positions of each column.
COLS = {
    "player": (0.015, 0.30),
    "kills": (0.355, 0.45),
    "assists": (0.445, 0.52),
    "damage": (0.515, 0.60),
    "survival": (0.595, 0.685),
    "hp_recovered": (0.685, 0.775),
    "rescues": (0.775, 0.845),
    "return": (0.845, 0.91),
    "score": (0.905, 0.995),
}


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


# ============================================================
# IMAGE / OCR HELPERS
# ============================================================

def resize_for_ocr(image, factor=2):
    """
    Moderate enlargement for OCR.
    2x is considerably faster than the previous 4x method.
    """

    return cv2.resize(
        image,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_CUBIC
    )


def clean_name(text):
    """
    Clean common OCR artifacts from player names.
    """

    text = text.strip()

    text = re.sub(
        r"[^A-Za-z0-9_*.\-]",
        "",
        text
    )

    return text.strip("._-")


def ocr_single(cell, psm=7, whitelist=None):
    """
    Single OCR pass.

    We deliberately do NOT run two OCR passes per cell.
    """

    if cell is None or cell.size == 0:
        return ""

    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )

    gray = resize_for_ocr(
        gray,
        2
    )

    config = f"--psm {psm}"

    if whitelist:
        config += (
            f" -c tessedit_char_whitelist={whitelist}"
        )

    text = pytesseract.image_to_string(
        gray,
        config=config
    )

    return text.strip()


# ============================================================
# NAME EXTRACTION
# ============================================================

def extract_player_name(cell):
    """
    Extract a single player name from one row.

    The crop is restricted to the player column so OCR
    cannot read the statistics belonging to another player.
    """

    if cell is None or cell.size == 0:
        return None

    # First attempt: normal grayscale OCR.
    text = ocr_single(
        cell,
        psm=7
    )

    name = clean_name(text)

    if (
        len(name) >= 2
        and not re.fullmatch(r"\d+", name)
    ):
        return name

    # Second attempt only if the first attempt failed.
    # This is intentionally used only for names that failed.
    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        threshold,
        config="--psm 7"
    ).strip()

    name = clean_name(text)

    if (
        len(name) >= 2
        and not re.fullmatch(r"\d+", name)
    ):
        return name

    return None


# ============================================================
# NUMBER PARSING
# ============================================================

def parse_number(text, kind):
    """
    Convert OCR text into a usable number.
    """

    if not text:
        return None

    text = (
        text.replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("l", "1")
            .replace(",", ".")
    )

    values = []

    if kind == "survival":

        matches = re.findall(
            r"\d{1,2}(?:\.\d+)?",
            text
        )

        for value in matches:

            try:
                number = float(value)

                if 0 < number <= 60:
                    values.append(number)

            except Exception:
                pass

    elif kind == "score":

        matches = re.findall(
            r"\d+(?:\.\d+)?",
            text
        )

        for value in matches:

            try:

                # PUBG scores are normally displayed with
                # a decimal point.
                if "." not in value and len(value) == 3:
                    value = (
                        value[:-1]
                        + "."
                        + value[-1]
                    )

                number = float(value)

                if 0 <= number <= 150:
                    values.append(number)

            except Exception:
                pass

    else:

        matches = re.findall(
            r"\d+",
            text
        )

        for value in matches:

            try:
                values.append(
                    int(value)
                )

            except Exception:
                pass

    if not values:
        return None

    return values[0]


# ============================================================
# ROW EXTRACTION
# ============================================================

def extract_row(table, row_center):
    """
    Extract exactly one player row.

    The row is processed independently.
    """

    h, w = table.shape[:2]

    # Row height.
    #
    # We use a relatively narrow vertical crop to prevent
    # neighboring players from entering the OCR.
    row_height = 0.075

    y_center = int(
        h * row_center
    )

    half_height = int(
        h * row_height / 2
    )

    y0 = max(
        0,
        y_center - half_height
    )

    y1 = min(
        h,
        y_center + half_height
    )

    # --------------------------------------------------------
    # PLAYER NAME
    # --------------------------------------------------------

    player_a, player_b = COLS["player"]

    player_cell = table[
        y0:y1,
        int(w * player_a):
        int(w * player_b)
    ]

    player = extract_player_name(
        player_cell
    )

    # If there is absolutely no readable player name,
    # consider this an empty row.
    if not player:
        return None

    result = {
        "player": player
    }

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    whitelist = "0123456789.,"

    for key, (a, b) in COLS.items():

        if key == "player":
            continue

        # Survival column
        if key == "survival":

            xa = int(
                w * 0.595
            )

            xb = int(
                w * 0.70
            )

            cell = table[
                y0:y1,
                xa:xb
            ]

            text = ocr_single(
                cell,
                psm=7,
                whitelist=whitelist
            )

            value = parse_number(
                text,
                "survival"
            )

        else:

            xa = int(
                w * a
            )

            xb = int(
                w * b
            )

            cell = table[
                y0:y1,
                xa:xb
            ]

            text = ocr_single(
                cell,
                psm=7,
                whitelist=whitelist
            )

            if key == "score":
                kind = "score"
            else:
                kind = "int"

            value = parse_number(
                text,
                kind
            )

        result[key] = value

    return result


# ============================================================
# METADATA
# ============================================================

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

    gray = resize_for_ocr(
        gray,
        1.5
    )

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    lower = text.lower()

    map_name = None

    for map_name_candidate in MAPS:

        if (
            map_name_candidate.lower()
            in lower
        ):
            map_name = map_name_candidate
            break

    # Placement is intentionally left editable.
    placement = None

    return placement, map_name


# ============================================================
# COMPLETE IMAGE ANALYSIS
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

    players = []

    # --------------------------------------------------------
    # IMPORTANT:
    # We ALWAYS inspect all 4 possible player positions.
    #
    # A match may contain:
    # 2 players
    # 3 players
    # 4 players
    #
    # We do NOT assume the same players appear in every match.
    # --------------------------------------------------------

    for row_center in ROW_Y:

        player = extract_row(
            table,
            row_center
        )

        if player:

            # Prevent duplicate names if OCR accidentally
            # reads the same player twice.
            duplicate = False

            for existing in players:

                if (
                    existing["player"].lower()
                    == player["player"].lower()
                ):
                    duplicate = True
                    break

            if not duplicate:

                player["match"] = match_number

                players.append(
                    player
                )

    placement, map_name = parse_metadata(
        img
    )

    return {
        "match": match_number,
        "map": map_name,
        "placement": placement,
        "players": players
    }


# ============================================================
# WEB PAGE
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


# ============================================================
# OCR API
# ============================================================

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

            result = analyze_image(
                data,
                i
            )

            results.append(
                result
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


# ============================================================
# EXPORT
# ============================================================

@app.post("/api/export")
async def export(
    payload: dict
):

    return {
        "ok": True
    }
