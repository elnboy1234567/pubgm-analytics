import re
import cv2
import pytesseract
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List

app = FastAPI(title="PUBGM Analytics MVP")

# Coordenadas de la tabla de resultados de PUBG Mobile
TABLE = (0.21, 0.47, 0.80, 0.695)

# Posiciones de respaldo de las filas
ROW_Y = [0.26, 0.445, 0.625, 0.81]

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


def upscale(im, factor=2):
    return cv2.resize(
        im,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_LINEAR
    )


def clean_name(s):
    s = re.sub(r"[^A-Za-z0-9_*.\-]", "", s)
    return s.strip("._-")


def ocr_text(cell, psm=7, whitelist=None):
    """
    Ejecuta un único OCR por celda para reducir muchísimo
    el tiempo de procesamiento.
    """

    if cell is None or cell.size == 0:
        return []

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    gray = upscale(gray, 2)

    cfg = f"--psm {psm}"

    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"

    text = pytesseract.image_to_string(
        gray,
        config=cfg
    ).strip()

    return [text] if text else []


def parse_number(candidates, kind):
    vals = []

    for t in candidates:

        t = (
            t.replace("O", "0")
             .replace("o", "0")
             .replace("I", "1")
             .replace("l", "1")
             .replace(",", ".")
        )

        if kind == "survival":

            for m in re.findall(
                r"\d{1,2}(?:\.\d+)?",
                t
            ):
                try:
                    v = float(m)

                    if 0 < v <= 60:
                        vals.append(v)

                except Exception:
                    pass

        elif kind == "score":

            for m in re.findall(
                r"\d+(?:\.\d+)?",
                t
            ):

                s = m

                if "." not in s and len(s) == 3:
                    s = s[:-1] + "." + s[-1]

                try:
                    v = float(s)

                    if 0 <= v <= 150:
                        vals.append(v)

                except Exception:
                    pass

        else:

            for m in re.findall(
                r"\d+",
                t
            ):
                try:
                    vals.append(int(m))
                except Exception:
                    pass

    if not vals:
        return None

    counts = {}

    for v in vals:
        counts[v] = counts.get(v, 0) + 1

    return max(
        counts,
        key=lambda v: (
            counts[v],
            -vals.index(v)
        )
    )


def detect_rows(table):
    """
    Detecta las filas de jugadores utilizando un único OCR
    sobre la tabla.
    """

    gray = cv2.cvtColor(
        table,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_LINEAR
    )

    df = pytesseract.image_to_data(
        gray,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    if not df or "conf" not in df:
        return [
            int(table.shape[0] * v)
            for v in ROW_Y
        ]

    ys = []

    total = len(df["text"])

    for i in range(total):

        text = str(df["text"][i]).strip()

        if not text:
            continue

        try:
            conf = float(df["conf"][i])
        except Exception:
            continue

        if conf < 25:
            continue

        left = float(df["left"][i])
        top = float(df["top"][i])
        height = float(df["height"][i])

        w = gray.shape[1]
        h = gray.shape[0]

        if left > w * 0.34 and top > h * 0.12:
            ys.append(
                top + height / 2
            )

    if not ys:
        return [
            int(table.shape[0] * v)
            for v in ROW_Y
        ]

    ys.sort()

    clusters = []

    for y in ys:

        if not clusters:
            clusters.append([y])

        elif y - clusters[-1][-1] > 30:
            clusters.append([y])

        else:
            clusters[-1].append(y)

    centers = [
        sum(c) / len(c)
        for c in clusters
        if len(c) >= 2
    ]

    # Volver de 1.5x al tamaño original
    centers = [
        c / 1.5
        for c in centers
    ]

    # Eliminar posible encabezado
    centers = [
        c
        for c in centers
        if c > table.shape[0] * 0.16
    ]

    if not centers or len(centers) > 8:
        return [
            int(table.shape[0] * v)
            for v in ROW_Y
        ]

    return centers[:8]


def extract_row(table, yc):

    h, w = table.shape[:2]

    y0 = max(
        0,
        int(yc - 24)
    )

    y1 = min(
        h,
        int(yc + 24)
    )

    out = {}

    # -------------------------
    # NOMBRE DEL JUGADOR
    # -------------------------

    a, b = COLS["player"]

    cell = table[
        y0:y1,
        int(w * a):int(w * b)
    ]

    names = []

    for t in ocr_text(
        cell,
        7
    ):

        n = clean_name(t)

        if (
            len(n) >= 3
            and not re.fullmatch(
                r"\d+",
                n
            )
        ):
            names.append(n)

    if not names:
        return None

    out["player"] = max(
        names,
        key=len
    )

    # -------------------------
    # ESTADÍSTICAS
    # -------------------------

    whitelist = "0123456789.,"

    for key, (a, b) in COLS.items():

        if key == "player":
            continue

        if key == "survival":

            xa = int(w * 0.595)
            xb = int(w * 0.70)

            cell = table[
                y0:y1,
                xa:xb
            ]

            candidates = ocr_text(
                cell,
                11,
                whitelist
            )

            kind = "survival"

        else:

            cell = table[
                y0:y1,
                int(w * a):int(w * b)
            ]

            candidates = ocr_text(
                cell,
                7,
                whitelist
            )

            kind = (
                "score"
                if key == "score"
                else "int"
            )

        out[key] = parse_number(
            candidates,
            kind
        )

    return out


def parse_metadata(img):

    h, w = img.shape[:2]

    header = img[
        int(0.07 * h):int(0.47 * h),
        int(0.20 * w):int(0.82 * w)
    ]

    gray = cv2.cvtColor(
        header,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_LINEAR
    )

    txt = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    low = txt.lower()

    map_name = None

    for m in MAPS:

        if m.lower() in low:
            map_name = m
            break

    # Se mantiene editable desde la interfaz
    placement = None

    return placement, map_name


def analyze_image(data, match_number):

    arr = np.frombuffer(
        data,
        np.uint8
    )

    img = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError(
            "No se pudo leer la imagen"
        )

    h, w = img.shape[:2]

    x0, y0, x1, y1 = TABLE

    table = img[
        int(y0 * h):int(y1 * h),
        int(x0 * w):int(x1 * w)
    ]

    rows = detect_rows(table)

    records = []

    for yc in rows:

        r = extract_row(
            table,
            yc
        )

        if r and r.get("player"):

            duplicate = any(
                x["player"].lower()
                == r["player"].lower()
                for x in records
            )

            if not duplicate:

                r["match"] = match_number

                records.append(r)

    placement, map_name = parse_metadata(
        img
    )

    return {
        "match": match_number,
        "map": map_name,
        "placement": placement,
        "players": records
    }


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

    for i, f in enumerate(files, 1):

        try:

            data = await f.read()

            result = analyze_image(
                data,
                i
            )

            results.append(result)

        except Exception as e:

            results.append(
                {
                    "match": i,
                    "error": str(e),
                    "players": []
                }
            )

    return JSONResponse(
        {
            "matches": results
        }
    )


@app.post("/api/export")
async def export(payload: dict):

    return {
        "ok": True
    }
