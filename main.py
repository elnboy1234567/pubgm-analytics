import re
import cv2
import numpy as np

from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

# ============================================================
# RAPIDOCR
# ============================================================

try:
    from rapidocr_onnxruntime import RapidOCR

    OCR_ENGINE = RapidOCR()
    RAPIDOCR_AVAILABLE = True

except Exception:
    OCR_ENGINE = None
    RAPIDOCR_AVAILABLE = False


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PUBGM Analytics"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_PLAYERS = 2
MAX_PLAYERS = 4

NORMALIZED_WIDTH = 1200
NORMALIZED_HEIGHT = 320

# Columnas relativas dentro de la tabla normalizada.
#
# PUBG mantiene prácticamente la misma estructura interna
# aunque cambie la resolución o el dispositivo.
#
# Se han definido como límites, NO como coordenadas de pantalla.

COLUMN_RATIOS = {
    "player": (0.00, 0.365),
    "kills": (0.365, 0.435),
    "assists": (0.435, 0.505),
    "damage": (0.505, 0.585),
    "survival": (0.585, 0.655),
    "hp_recovered": (0.655, 0.755),
    "rescues": (0.755, 0.835),
    "return": (0.835, 0.915),
    "score": (0.915, 1.000),
}


# ============================================================
# UTILIDADES
# ============================================================

def clean_spaces(text):

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    return text


def numeric_cleanup(text):

    if not text:
        return ""

    text = str(text)

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
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    return text


# ============================================================
# RAPIDOCR
# ============================================================

def run_ocr(image):

    if not RAPIDOCR_AVAILABLE:
        return []

    if image is None:
        return []

    if image.size == 0:
        return []

    try:

        result, _ = OCR_ENGINE(
            image
        )

    except Exception:
        return []

    if not result:
        return []

    output = []

    for item in result:

        try:

            box = item[0]
            text = clean_spaces(
                item[1]
            )
            confidence = float(
                item[2]
            )

        except Exception:
            continue

        if not text:
            continue

        if confidence < 0.25:
            continue

        xs = [
            int(point[0])
            for point in box
        ]

        ys = [
            int(point[1])
            for point in box
        ]

        x0 = min(xs)
        x1 = max(xs)
        y0 = min(ys)
        y1 = max(ys)

        output.append(
            {
                "text": text,
                "confidence": confidence,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "cx": (x0 + x1) / 2,
                "cy": (y0 + y1) / 2,
            }
        )

    return output


# ============================================================
# DETECTAR PANEL DE RESULTADOS
# ============================================================

def detect_result_panel(image):

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Buscamos zonas oscuras grandes.
    # La tabla de PUBG es una de las zonas más oscuras
    # y estructuradas de la pantalla.
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    dark = cv2.threshold(
        blurred,
        105,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 9)
    )

    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        dark,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        area = cw * ch

        if area < w * h * 0.04:
            continue

        ratio = cw / max(
            ch,
            1
        )

        # El panel es claramente horizontal.
        if ratio < 1.5:
            continue

        if cw < w * 0.45:
            continue

        candidates.append(
            (
                area,
                x,
                y,
                cw,
                ch
            )
        )

    if candidates:

        # Preferimos la región grande y horizontal.
        candidates.sort(
            reverse=True
        )

        _, x, y, cw, ch = candidates[0]

        # Añadir margen.
        pad_x = int(cw * 0.01)
        pad_y = int(ch * 0.01)

        x0 = max(
            0,
            x - pad_x
        )

        y0 = max(
            0,
            y - pad_y
        )

        x1 = min(
            w,
            x + cw + pad_x
        )

        y1 = min(
            h,
            y + ch + pad_y
        )

        return (
            x0,
            y0,
            x1,
            y1
        )

    return None


# ============================================================
# ENCONTRAR ENCABEZADO
# ============================================================

HEADER_WORDS = [
    "name",
    "nombre",
    "eliminations",
    "eliminaciones",
    "assists",
    "asistencias",
    "damage",
    "daño",
    "dano",
    "survived",
    "supervivencia",
    "rating",
    "puntaje",
]


def header_score(
    text
):

    text = normalize_text(
        text
    )

    best = 0

    for word in HEADER_WORDS:

        if text == word:
            best = max(
                best,
                1.0
            )
            continue

        if word in text:
            best = max(
                best,
                0.75
            )

    return best


def find_header(
    ocr,
    panel
):

    px0, py0, px1, py1 = panel

    candidates = []

    for item in ocr:

        if item["cx"] < px0:
            continue

        if item["cx"] > px1:
            continue

        if item["cy"] < py0:
            continue

        if item["cy"] > py1:
            continue

        score = header_score(
            item["text"]
        )

        if score <= 0:
            continue

        candidates.append(
            (
                score,
                item
            )
        )

    if not candidates:
        return None

    # Preferimos "Nombre"/"Name".
    name_candidates = []

    for score, item in candidates:

        text = normalize_text(
            item["text"]
        )

        if (
            text == "name"
            or
            text == "nombre"
        ):
            name_candidates.append(
                (
                    score,
                    item
                )
            )

    if name_candidates:

        name_candidates.sort(
            key=lambda x: x[1]["cy"]
        )

        return name_candidates[0][1]

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0][1]


# ============================================================
# NORMALIZAR PANEL
# ============================================================

def normalize_panel(
    image,
    panel
):

    x0, y0, x1, y1 = panel

    crop = image[
        y0:y1,
        x0:x1
    ]

    if crop.size == 0:
        return None

    normalized = cv2.resize(
        crop,
        (
            NORMALIZED_WIDTH,
            NORMALIZED_HEIGHT
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return normalized


# ============================================================
# DETECTAR FILAS
# ============================================================

def detect_rows(
    normalized
):

    gray = cv2.cvtColor(
        normalized,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # El encabezado es negro.
    # Las filas tienen texto claro.
    #
    # Buscamos concentración horizontal de píxeles claros.
    # --------------------------------------------------------

    bright = cv2.threshold(
        gray,
        160,
        255,
        cv2.THRESH_BINARY
    )[1]

    # Ignorar aproximadamente el encabezado.
    search = bright[
        int(NORMALIZED_HEIGHT * 0.28):
        int(NORMALIZED_HEIGHT * 0.98),
        :
    ]

    projection = np.sum(
        search > 0,
        axis=1
    )

    # Suavizar.
    kernel = np.ones(
        7,
        dtype=np.float32
    ) / 7

    projection = np.convolve(
        projection,
        kernel,
        mode="same"
    )

    # Umbral adaptativo.
    threshold = max(
        np.percentile(
            projection,
            65
        ),
        10
    )

    active = projection > threshold

    groups = []

    start = None

    for i, value in enumerate(
        active
    ):

        if value and start is None:

            start = i

        elif (
            not value
            and
            start is not None
        ):

            if i - start >= 3:

                groups.append(
                    (
                        start,
                        i
                    )
                )

            start = None

    if start is not None:

        groups.append(
            (
                start,
                len(active)
            )
        )

    centers = []

    offset = int(
        NORMALIZED_HEIGHT * 0.28
    )

    for start, end in groups:

        center = int(
            (
                start
                + end
            ) / 2
        )

        center += offset

        centers.append(
            center
        )

    # --------------------------------------------------------
    # Agrupar centros demasiado cercanos.
    # --------------------------------------------------------

    filtered = []

    for center in centers:

        if not filtered:

            filtered.append(
                center
            )

            continue

        if (
            center
            - filtered[-1]
            < 22
        ):

            filtered[-1] = int(
                (
                    filtered[-1]
                    + center
                ) / 2
            )

        else:

            filtered.append(
                center
            )

    # --------------------------------------------------------
    # Normalmente tenemos 4 jugadores.
    # Si hay ruido, nos quedamos con los centros
    # más regularmente separados.
    # --------------------------------------------------------

    if len(filtered) > MAX_PLAYERS:

        candidates = []

        for i in range(
            len(filtered)
            - MAX_PLAYERS
            + 1
        ):

            subset = filtered[
                i:i + MAX_PLAYERS
            ]

            gaps = np.diff(
                subset
            )

            if len(gaps) == 0:
                continue

            variation = np.std(
                gaps
            )

            candidates.append(
                (
                    variation,
                    subset
                )
            )

        if candidates:

            candidates.sort(
                key=lambda x: x[0]
            )

            filtered = candidates[0][1]

    if (
        len(filtered)
        < MIN_PLAYERS
    ):
        return []

    return filtered


# ============================================================
# CORTAR CELDA
# ============================================================

def crop_cell(
    table,
    row_center,
    column
):

    x0 = int(
        NORMALIZED_WIDTH
        * column[0]
    )

    x1 = int(
        NORMALIZED_WIDTH
        * column[1]
    )

    # --------------------------------------------------------
    # Cada fila tiene margen vertical.
    # --------------------------------------------------------

    row_height = 47

    y0 = max(
        0,
        int(
            row_center
            - row_height / 2
        )
    )

    y1 = min(
        NORMALIZED_HEIGHT,
        int(
            row_center
            + row_height / 2
        )
    )

    cell = table[
        y0:y1,
        x0:x1
    ]

    return cell


# ============================================================
# LEER NOMBRE
# ============================================================

def clean_player(
    text
):

    if not text:
        return ""

    text = clean_spaces(
        text
    )

    # El OCR puede devolver elementos
    # del borde de la celda.
    text = text.strip(
        "|[]{}()"
    )

    # Mantener caracteres habituales
    # de nombres PUBG.
    text = re.sub(
        r"[^A-Za-z0-9_.*\-]+",
        "",
        text
    )

    return text


def read_player(
    cell
):

    ocr = run_ocr(
        cell
    )

    candidates = []

    for item in ocr:

        text = clean_player(
            item["text"]
        )

        if len(text) < 3:
            continue

        # No aceptar únicamente números.
        if re.fullmatch(
            r"\d+",
            text
        ):
            continue

        candidates.append(
            (
                item["confidence"],
                text
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0][1]


# ============================================================
# LEER ENTERO
# ============================================================

def read_integer(
    cell,
    minimum=0,
    maximum=9999
):

    ocr = run_ocr(
        cell
    )

    values = []

    for item in ocr:

        text = numeric_cleanup(
            item["text"]
        )

        matches = re.findall(
            r"\d+",
            text
        )

        for match in matches:

            try:
                value = int(
                    match
                )
            except Exception:
                continue

            if (
                minimum
                <= value
                <= maximum
            ):

                values.append(
                    (
                        item["confidence"],
                        value
                    )
                )

    if not values:
        return None

    # Mejor confianza primero.
    values.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return values[0][1]


# ============================================================
# LEER PUNTUACIÓN
# ============================================================

def read_score(
    cell
):

    ocr = run_ocr(
        cell
    )

    values = []

    for item in ocr:

        text = numeric_cleanup(
            item["text"]
        )

        matches = re.findall(
            r"\d+(?:[.,]\d+)?",
            text
        )

        for match in matches:

            try:

                value = float(
                    match.replace(
                        ",",
                        "."
                    )
                )

            except Exception:
                continue

            if 0 <= value <= 150:

                values.append(
                    (
                        item["confidence"],
                        value
                    )
                )

    if not values:
        return None

    values.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return values[0][1]


# ============================================================
# LEER SUPERVIVENCIA
# ============================================================

def read_survival(
    cell
):

    ocr = run_ocr(
        cell
    )

    values = []

    for item in ocr:

        text = numeric_cleanup(
            item["text"]
        )

        matches = re.findall(
            r"(\d{1,2})[:.](\d{1,2})",
            text
        )

        for minutes, seconds in matches:

            try:

                m = int(
                    minutes
                )

                s = int(
                    seconds
                )

                if (
                    0 <= m <= 60
                    and
                    0 <= s < 60
                ):

                    values.append(
                        (
                            item["confidence"],
                            f"{m}:{s:02d}"
                        )
                    )

            except Exception:
                pass

    if not values:
        return None

    values.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return values[0][1]


# ============================================================
# EXTRAER JUGADOR
# ============================================================

def extract_row(
    table,
    row_center
):

    result = {
        "player": None,
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
    # NOMBRE
    # --------------------------------------------------------

    result["player"] = read_player(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["player"]
        )
    )

    # --------------------------------------------------------
    # KILLS
    # --------------------------------------------------------

    result["kills"] = read_integer(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["kills"]
        ),
        0,
        50
    )

    # --------------------------------------------------------
    # ASSISTS
    # --------------------------------------------------------

    result["assists"] = read_integer(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["assists"]
        ),
        0,
        50
    )

    # --------------------------------------------------------
    # DAMAGE
    # --------------------------------------------------------

    result["damage"] = read_integer(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["damage"]
        ),
        0,
        5000
    )

    # --------------------------------------------------------
    # SURVIVAL
    # --------------------------------------------------------

    result["survival"] = read_survival(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["survival"]
        )
    )

    # --------------------------------------------------------
    # HP RECOVERED
    # --------------------------------------------------------

    result["hp_recovered"] = read_integer(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["hp_recovered"]
        ),
        0,
        5000
    )

    # --------------------------------------------------------
    # RESCUES
    # --------------------------------------------------------

    result["rescues"] = read_integer(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["rescues"]
        ),
        0,
        50
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    result["return"] = read_integer(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["return"]
        ),
        0,
        50
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    result["score"] = read_score(
        crop_cell(
            table,
            row_center,
            COLUMN_RATIOS["score"]
        )
    )

    return result


# ============================================================
# MAPA
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


def detect_map(
    ocr
):

    text = " ".join(
        item["text"]
        for item in ocr
    )

    text = normalize_text(
        text
    )

    for game_map in MAPS:

        if normalize_text(
            game_map
        ) in text:

            return game_map

    return None


# ============================================================
# ANALIZAR IMAGEN
# ============================================================

def analyze_image(
    data,
    match_number
):

    if not RAPIDOCR_AVAILABLE:

        return {
            "match": match_number,
            "error": (
                "RapidOCR no está instalado. "
                "Hay que actualizar requirements.txt."
            ),
            "players": []
        }

    array = np.frombuffer(
        data,
        np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "No se pudo leer la imagen"
        )

    original_h, original_w = image.shape[:2]

    # --------------------------------------------------------
    # 1. OCR GLOBAL
    #
    # Solamente para localizar el panel/estructura.
    # --------------------------------------------------------

    global_ocr = run_ocr(
        image
    )

    # --------------------------------------------------------
    # 2. LOCALIZAR PANEL
    # --------------------------------------------------------

    panel = detect_result_panel(
        image
    )

    if panel is None:

        return {
            "match": match_number,
            "error": (
                "No se pudo localizar "
                "el panel de resultados."
            ),
            "players": [],
            "debug": {
                "image_size": [
                    original_w,
                    original_h
                ],
                "ocr_words": len(
                    global_ocr
                )
            }
        }

    # --------------------------------------------------------
    # 3. NORMALIZAR
    # --------------------------------------------------------

    normalized = normalize_panel(
        image,
        panel
    )

    if normalized is None:

        return {
            "match": match_number,
            "error": (
                "No se pudo normalizar "
                "el panel."
            ),
            "players": []
        }

    # --------------------------------------------------------
    # 4. DETECTAR FILAS
    # --------------------------------------------------------

    rows = detect_rows(
        normalized
    )

    if not rows:

        return {
            "match": match_number,
            "error": (
                "No se pudieron detectar "
                "las filas de jugadores."
            ),
            "players": [],
            "debug": {
                "panel": panel,
                "normalized_size": [
                    NORMALIZED_WIDTH,
                    NORMALIZED_HEIGHT
                ]
            }
        }

    # --------------------------------------------------------
    # 5. EXTRAER JUGADORES
    # --------------------------------------------------------

    players = []

    for row_index, row_center in enumerate(
        rows,
        1
    ):

        player = extract_row(
            normalized,
            row_center
        )

        player["row"] = row_index

        # ----------------------------------------------------
        # IMPORTANTE:
        #
        # No descartamos automáticamente una fila solo
        # porque el nombre falle.
        #
        # Esto evita perder jugadores.
        # ----------------------------------------------------

        if any(
            value is not None
            for key, value in player.items()
            if key != "row"
        ):

            players.append(
                player
            )

    # --------------------------------------------------------
    # 6. MAPA
    # --------------------------------------------------------

    map_name = detect_map(
        global_ocr
    )

    # --------------------------------------------------------
    # 7. RESULTADO
    # --------------------------------------------------------

    return {
        "match": match_number,
        "map": map_name,
        "placement": None,
        "players": players,

        "debug": {
            "image_size": [
                original_w,
                original_h
            ],
            "panel": panel,
            "normalized_size": [
                NORMALIZED_WIDTH,
                NORMALIZED_HEIGHT
            ],
            "rows": rows,
            "ocr_engine": "RapidOCR",
            "players_detected": len(
                players
            )
        }
    }


# ============================================================
# HOME
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
# API ANALYZE
# ============================================================

@app.post(
    "/api/analyze"
)
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

@app.post(
    "/api/export"
)
async def export(
    payload: dict
):

    return {
        "ok": True
    }
