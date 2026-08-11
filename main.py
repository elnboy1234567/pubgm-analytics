import re
import cv2
import pytesseract
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
app = FastAPI(title="PUBGM Analytics")
# ============================================================
# CONFIGURACIÓN DE LA TABLA
# ============================================================
TABLE = (0.21, 0.47, 0.80, 0.695)
# Coordenadas reales obtenidas de la captura de diagnóstico.
# Son coordenadas dentro del recorte TABLE.
ROW_CENTERS = [10, 90, 172, 254]
# Límites horizontales de cada columna.
#
# Se dejan márgenes entre columnas para evitar que el OCR
# capture números de la columna siguiente.
#
COLUMN_RANGES = {
    "player":      (20, 400),
    "kills":       (620, 720),
    "assists":     (750, 850),
    "damage":      (855, 955),
    "survival":    (960, 1095),
    "hp_recovered":(1100, 1215),
    "rescues":     (1235, 1315),
    "return":      (1345, 1430),
    "score":       (1450, 1555),
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
def clean_player(text):
    text = clean_spaces(text)
    # Eliminar caracteres de borde.
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
    # Mantener caracteres habituales en nombres PUBG.
    text = re.sub(
        r"[^A-Za-z0-9_*.\-]",
        "",
        text
    )
    return text.strip()
# ============================================================
# PREPROCESADO OCR
# ============================================================
def prepare_variants(cell):
    variants = []
    # Escala original.
    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )
    # Ampliación importante porque los números de la tabla
    # son relativamente pequeños.
    up = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )
    variants.append(up)
    # Contraste.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(
        up
    )
    variants.append(enhanced)
    # Otsu.
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )
    variants.append(otsu)
    # Invertido.
    inverted = cv2.bitwise_not(
        otsu
    )
    variants.append(inverted)
    return variants
# ============================================================
# OCR DE CELDA
# ============================================================
def read_cell(
    cell,
    kind
):
    if cell is None:
        return []
    if cell.size == 0:
        return []
    variants = prepare_variants(
        cell
    )
    candidates = []
    if kind == "player":
        configs = [
            "--psm 7",
            "--psm 8",
            "--psm 13",
        ]
    elif kind == "survival":
        configs = [
            "--psm 7",
            "--psm 8",
            "--psm 13",
        ]
    elif kind == "score":
        configs = [
            "--psm 7",
            "--psm 8",
            "--psm 13",
        ]
    else:
        configs = [
            "--psm 7",
            "--psm 8",
            "--psm 13",
        ]
    for image in variants:
        for config in configs:
            if kind == "player":
                text = pytesseract.image_to_string(
                    image,
                    config=config
                )
            elif kind == "survival":
                text = pytesseract.image_to_string(
                    image,
                    config=config +
                    " -c tessedit_char_whitelist=0123456789:."
                )
            else:
                text = pytesseract.image_to_string(
                    image,
                    config=config +
                    " -c tessedit_char_whitelist=0123456789.,"
                )
            text = clean_spaces(
                text
            )
            if text:
                candidates.append(
                    text
                )
    return candidates
# ============================================================
# NÚMEROS
# ============================================================
def number_clean(text):
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
    return text
def extract_integer(
    candidates,
    minimum=0,
    maximum=9999
):
    values = []
    for candidate in candidates:
        candidate = number_clean(
            candidate
        )
        matches = re.findall(
            r"\d+",
            candidate
        )
        for match in matches:
            try:
                value = int(
                    match
                )
                if (
                    minimum
                    <= value
                    <= maximum
                ):
                    values.append(
                        value
                    )
            except Exception:
                pass
    if not values:
        return None
    # Votación.
    counts = {}
    for value in values:
        counts[value] = (
            counts.get(
                value,
                0
            ) + 1
        )
    return max(
        counts,
        key=lambda value:
        counts[value]
    )
def extract_score(
    candidates
):
    values = []
    for candidate in candidates:
        candidate = number_clean(
            candidate
        )
        matches = re.findall(
            r"\d+(?:\.\d+)?",
            candidate
        )
        for match in matches:
            try:
                value = float(
                    match
                )
                if (
                    0 <= value <= 150
                ):
                    values.append(
                        value
                    )
            except Exception:
                pass
    if not values:
        return None
    counts = {}
    for value in values:
        counts[value] = (
            counts.get(
                value,
                0
            ) + 1
        )
    return max(
        counts,
        key=lambda value:
        counts[value]
    )
def extract_survival(
    candidates
):
    values = []
    for candidate in candidates:
        candidate = number_clean(
            candidate
        )
        # Formato habitual: 16:42
        matches = re.findall(
            r"(\d{1,2})[:.](\d{1,2})",
            candidate
        )
        for minutes, seconds in matches:
            try:
                m = int(minutes)
                s = int(seconds)
                if (
                    0 <= m <= 60
                    and
                    0 <= s < 60
                ):
                    values.append(
                        m + s / 100
                    )
            except Exception:
                pass
        # Algunos OCR convierten "min" en texto
        # y dejan únicamente el valor.
        if not matches:
            simple = re.findall(
                r"\d+(?:\.\d+)?",
                candidate
            )
            for value in simple:
                try:
                    number = float(
                        value
                    )
                    if (
                        0 < number <= 60
                    ):
                        values.append(
                            number
                        )
                except Exception:
                    pass
    if not values:
        return None
    counts = {}
    for value in values:
        rounded = round(
            value,
            2
        )
        counts[rounded] = (
            counts.get(
                rounded,
                0
            ) + 1
        )
    return max(
        counts,
        key=lambda value:
        counts[value]
    )
# ============================================================
# NOMBRE
# ============================================================
def extract_player(
    candidates
):
    valid = []
    for candidate in candidates:
        name = clean_player(
            candidate
        )
        if len(name) < 3:
            continue
        if re.fullmatch(
            r"\d+",
            name
        ):
            continue
        valid.append(
            name
        )
    if not valid:
        return None
    # La mayoría de las lecturas suelen coincidir.
    counts = {}
    for name in valid:
        key = name.lower()
        counts[key] = (
            counts.get(
                key,
                0
            ) + 1
        )
    best_key = max(
        counts,
        key=lambda key:
        counts[key]
    )
    # Devolver la versión original.
    for name in valid:
        if name.lower() == best_key:
            return name
    return None
# ============================================================
# DETECCIÓN DE FILAS
# ============================================================
def find_row_centers(table):
    # Como conocemos la estructura exacta de la pantalla,
    # primero intentamos localizar las cuatro posiciones reales.
    #
    # Se permite que se desplacen ligeramente.
    gray = cv2.cvtColor(
        table,
        cv2.COLOR_BGR2GRAY
    )
    height = gray.shape[0]
    # Las posiciones conocidas son:
    # 10, 90, 172, 254.
    #
    # Para una captura con las mismas proporciones,
    # estas posiciones son extremadamente estables.
    centers = []
    for center in ROW_CENTERS:
        if (
            center >= 0
            and
            center < height
        ):
            centers.append(
                center
            )
    return centers
# ============================================================
# EXTRAER UNA FILA
# ============================================================
def extract_row(
    table,
    center
):
    h, w = table.shape[:2]
    # Cada fila tiene aproximadamente 50 px de altura.
    y0 = max(
        0,
        int(center - 24)
    )
    y1 = min(
        h,
        int(center + 24)
    )
    row = table[
        y0:y1,
        :
    ]
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
    x0, x1 = COLUMN_RANGES[
        "player"
    ]
    cell = row[
        :,
        x0:x1
    ]
    candidates = read_cell(
        cell,
        "player"
    )
    result["player"] = extract_player(
        candidates
    )
    # Si no hay nombre, probablemente no hay jugador.
    if not result["player"]:
        return None
    # --------------------------------------------------------
    # KILLS
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "kills"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["kills"] = extract_integer(
        read_cell(
            cell,
            "integer"
        ),
        0,
        50
    )
    # --------------------------------------------------------
    # ASSISTS
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "assists"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["assists"] = extract_integer(
        read_cell(
            cell,
            "integer"
        ),
        0,
        50
    )
    # --------------------------------------------------------
    # DAMAGE
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "damage"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["damage"] = extract_integer(
        read_cell(
            cell,
            "integer"
        ),
        0,
        5000
    )
    # --------------------------------------------------------
    # SURVIVAL
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "survival"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["survival"] = extract_survival(
        read_cell(
            cell,
            "survival"
        )
    )
    # --------------------------------------------------------
    # HP RECOVERED
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "hp_recovered"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["hp_recovered"] = extract_integer(
        read_cell(
            cell,
            "integer"
        ),
        0,
        5000
    )
    # --------------------------------------------------------
    # RESCUES
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "rescues"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["rescues"] = extract_integer(
        read_cell(
            cell,
            "integer"
        ),
        0,
        50
    )
    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "return"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["return"] = extract_integer(
        read_cell(
            cell,
            "integer"
        ),
        0,
        50
    )
    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------
    x0, x1 = COLUMN_RANGES[
        "score"
    ]
    cell = row[
        :,
        x0:x1
    ]
    result["score"] = extract_score(
        read_cell(
            cell,
            "score"
        )
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
        "map": map_name,
        "placement": None
    }
# ============================================================
# ANALIZAR IMAGEN
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
    row_centers = find_row_centers(
        table
    )
    players = []
    for center in row_centers:
        result = extract_row(
            table,
            center
        )
        if result is None:
            continue
        # Evitar duplicados.
        duplicate = False
        for existing in players:
            if (
                existing["player"].lower()
                ==
                result["player"].lower()
            ):
                duplicate = True
                break
        if duplicate:
            continue
        result["match"] = match_number
        players.append(
            result
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
@app.post("/api/export")
async def export(
    payload: dict
):
    return {
        "ok": True
    }
