import re
import cv2
import unicodedata
import pytesseract
import numpy as np

from difflib import SequenceMatcher
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse


app = FastAPI(title="PUBGM Analytics")


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MAX_DETECTION_WIDTH = 1800

MIN_PLAYERS = 2
MAX_PLAYERS = 4

EXPECTED_COLUMNS_ES = [
    "nombre",
    "eliminaciones",
    "asistencias",
    "daño",
    "supervivencia",
    "vida recup",
    "rescates",
    "regresar",
    "puntaje",
]

EXPECTED_COLUMNS_EN = [
    "name",
    "eliminations",
    "assists",
    "damage",
    "survived",
    "health restored",
    "rescue",
    "recall",
    "rating",
]


# ============================================================
# UTILIDADES
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9 ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_spaces(text):
    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def resize_for_detection(img):
    h, w = img.shape[:2]

    if w <= MAX_DETECTION_WIDTH:
        return img, 1.0

    scale = MAX_DETECTION_WIDTH / w

    resized = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# OCR GENERAL
# ============================================================

def ocr_data(
    image,
    psm=11,
    whitelist=None
):
    config = f"--oem 3 --psm {psm}"

    if whitelist:
        config += (
            " -c tessedit_char_whitelist="
            + whitelist
        )

    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    results = []

    total = len(
        data["text"]
    )

    for i in range(total):

        text = clean_spaces(
            data["text"][i]
        )

        if not text:
            continue

        try:
            confidence = float(
                data["conf"][i]
            )
        except Exception:
            confidence = 0

        if confidence < 15:
            continue

        results.append(
            {
                "text": text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "cx": int(
                    data["left"][i]
                    + data["width"][i] / 2
                ),
                "cy": int(
                    data["top"][i]
                    + data["height"][i] / 2
                ),
                "confidence": confidence,
            }
        )

    return results


# ============================================================
# LOCALIZAR ENCABEZADO
# ============================================================

def find_header(
    img,
    ocr
):
    candidates = []

    for item in ocr:

        text = normalize_text(
            item["text"]
        )

        if not text:
            continue

        score_es = similarity(
            text,
            "nombre"
        )

        score_en = similarity(
            text,
            "name"
        )

        score = max(
            score_es,
            score_en
        )

        if score >= 0.55:

            candidates.append(
                (
                    score,
                    item
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best = candidates[0][1]

    return {
        "x": best["x"],
        "y": best["y"],
        "w": best["w"],
        "h": best["h"],
        "cx": best["cx"],
        "cy": best["cy"],
    }


# ============================================================
# LOCALIZAR COLUMNAS
# ============================================================

def find_columns(
    ocr,
    header
):

    header_y = header["cy"]

    possible_labels = []

    all_labels = (
        EXPECTED_COLUMNS_ES
        + EXPECTED_COLUMNS_EN
    )

    for item in ocr:

        # Los encabezados tienen que estar
        # aproximadamente en la misma altura.
        if abs(
            item["cy"] - header_y
        ) > 30:
            continue

        text = normalize_text(
            item["text"]
        )

        if not text:
            continue

        best_label = None
        best_score = 0

        for label in all_labels:

            score = similarity(
                text,
                label
            )

            if score > best_score:
                best_score = score
                best_label = label

        if best_score >= 0.52:

            possible_labels.append(
                {
                    "label": best_label,
                    "x": item["cx"],
                    "score": best_score,
                }
            )

    # --------------------------------------------------------
    # "Health Restored" / "Vida Recuperada"
    # puede aparecer como dos palabras.
    # --------------------------------------------------------

    if len(possible_labels) < 4:
        return None

    # Eliminar duplicados muy cercanos.
    columns = []

    for item in sorted(
        possible_labels,
        key=lambda x: x["x"]
    ):

        duplicate = False

        for existing in columns:

            if abs(
                item["x"]
                - existing["x"]
            ) < 20:

                if (
                    item["score"]
                    >
                    existing["score"]
                ):
                    existing.update(
                        item
                    )

                duplicate = True
                break

        if not duplicate:
            columns.append(
                item
            )

    columns.sort(
        key=lambda x: x["x"]
    )

    return columns


# ============================================================
# LOCALIZAR PANEL
# ============================================================

def estimate_table_bounds(
    img,
    header,
    columns
):

    h, w = img.shape[:2]

    header_y = header["cy"]

    column_x = [
        c["x"]
        for c in columns
    ]

    if not column_x:
        return None

    # La primera columna suele empezar
    # bastante antes del centro de "Nombre".
    left = max(
        0,
        int(
            min(column_x)
            - 80
        )
    )

    right = min(
        w,
        int(
            max(column_x)
            + 100
        )
    )

    # Estimación inicial.
    #
    # No es una coordenada fija:
    # depende de la posición del encabezado.
    top = max(
        0,
        int(
            header_y - 35
        )
    )

    bottom = min(
        h,
        int(
            header_y + 260
        )
    )

    return (
        left,
        top,
        right,
        bottom
    )


# ============================================================
# DETECCIÓN DE FILAS
# ============================================================

def detect_row_centers(
    img,
    bounds,
    ocr,
    header
):

    left, top, right, bottom = bounds

    header_y = header["cy"]

    # ========================================================
    # PRIMERA FUENTE:
    # palabras OCR debajo del encabezado
    # ========================================================

    y_points = []

    for item in ocr:

        if item["cy"] <= header_y + 20:
            continue

        if item["cy"] >= bottom:
            continue

        if item["cx"] < left:
            continue

        if item["cx"] > right:
            continue

        y_points.append(
            item["cy"]
        )

    if not y_points:
        return []

    y_points.sort()

    # ========================================================
    # AGRUPACIÓN POR ALTURA
    # ========================================================

    groups = []

    current = [
        y_points[0]
    ]

    for y in y_points[1:]:

        if abs(
            y - np.mean(current)
        ) <= 18:

            current.append(y)

        else:

            groups.append(
                current
            )

            current = [y]

    groups.append(
        current
    )

    centers = []

    for group in groups:

        if len(group) < 1:
            continue

        center = int(
            np.median(group)
        )

        # No aceptar cosas demasiado
        # cercanas al encabezado.
        if center <= header_y + 25:
            continue

        centers.append(
            center
        )

    # ========================================================
    # ELIMINAR GRUPOS DEMASIADO CERCANOS
    # ========================================================

    filtered = []

    for center in centers:

        if not filtered:
            filtered.append(
                center
            )
            continue

        if abs(
            center
            - filtered[-1]
        ) < 25:

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

    # ========================================================
    # LIMITAR A 2-4 JUGADORES
    # ========================================================

    if len(filtered) > MAX_PLAYERS:

        # Normalmente los primeros grupos
        # después del header son jugadores.
        filtered = filtered[
            :MAX_PLAYERS
        ]

    if len(filtered) < MIN_PLAYERS:
        return []

    return filtered


# ============================================================
# DETECTAR CELDAS
# ============================================================

def build_column_boundaries(
    columns,
    table_left,
    table_right
):

    columns = sorted(
        columns,
        key=lambda x: x["x"]
    )

    centers = [
        c["x"]
        for c in columns
    ]

    if not centers:
        return []

    boundaries = []

    boundaries.append(
        table_left
    )

    for i in range(
        len(centers) - 1
    ):

        midpoint = int(
            (
                centers[i]
                + centers[i + 1]
            ) / 2
        )

        boundaries.append(
            midpoint
        )

    boundaries.append(
        table_right
    )

    result = []

    for i, column in enumerate(
        columns
    ):

        result.append(
            {
                "label": column["label"],
                "x0": boundaries[i],
                "x1": boundaries[i + 1],
                "center": column["x"],
            }
        )

    return result


# ============================================================
# OCR DE UNA CELDA
# ============================================================

def prepare_cell(
    cell
):

    if cell is None:
        return []

    if cell.size == 0:
        return []

    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )

    # Escalado proporcional.
    up = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Contraste.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        up
    )

    return [
        up,
        enhanced
    ]


def read_cell(
    cell,
    numeric=False
):

    variants = prepare_cell(
        cell
    )

    if not variants:
        return []

    results = []

    for image in variants:

        if numeric:

            text = pytesseract.image_to_string(
                image,
                config=(
                    "--oem 3 --psm 7 "
                    "-c "
                    "tessedit_char_whitelist="
                    "0123456789:.,"
                )
            )

        else:

            text = pytesseract.image_to_string(
                image,
                config="--oem 3 --psm 7"
            )

        text = clean_spaces(
            text
        )

        if text:
            results.append(
                text
            )

    return results


# ============================================================
# LIMPIEZA DE NOMBRES
# ============================================================

def clean_player(
    text
):

    if not text:
        return ""

    text = clean_spaces(
        text
    )

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

    # Mantener caracteres habituales
    # en nombres PUBG.
    text = re.sub(
        r"[^A-Za-z0-9_*.\-]",
        "",
        text
    )

    return text.strip()


def choose_player(
    candidates
):

    valid = []

    for candidate in candidates:

        name = clean_player(
            candidate
        )

        if len(name) < 3:
            continue

        # Un nombre compuesto únicamente
        # por números probablemente no es válido.
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

    # Votación por coincidencia.
    scores = {}

    for name in valid:

        key = name.lower()

        scores[key] = (
            scores.get(
                key,
                0
            )
            + 1
        )

    best = max(
        scores,
        key=scores.get
    )

    for name in valid:

        if name.lower() == best:
            return name

    return None


# ============================================================
# LIMPIEZA NUMÉRICA
# ============================================================

def clean_number_text(
    text
):

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


def choose_integer(
    candidates,
    minimum=0,
    maximum=9999
):

    values = []

    for candidate in candidates:

        candidate = clean_number_text(
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
            except Exception:
                continue

            if (
                minimum
                <= value
                <= maximum
            ):
                values.append(
                    value
                )

    if not values:
        return None

    counts = {}

    for value in values:

        counts[value] = (
            counts.get(
                value,
                0
            )
            + 1
        )

    return max(
        counts,
        key=counts.get
    )


def choose_score(
    candidates
):

    values = []

    for candidate in candidates:

        candidate = clean_number_text(
            candidate
        )

        matches = re.findall(
            r"\d+(?:[.,]\d+)?",
            candidate
        )

        for match in matches:

            match = match.replace(
                ",",
                "."
            )

            try:
                value = float(
                    match
                )
            except Exception:
                continue

            if 0 <= value <= 150:
                values.append(
                    value
                )

    if not values:
        return None

    counts = {}

    for value in values:

        rounded = round(
            value,
            1
        )

        counts[rounded] = (
            counts.get(
                rounded,
                0
            )
            + 1
        )

    return max(
        counts,
        key=counts.get
    )


def choose_survival(
    candidates
):

    values = []

    for candidate in candidates:

        candidate = clean_number_text(
            candidate
        )

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
                        f"{m}:{s:02d}"
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
            )
            + 1
        )

    return max(
        counts,
        key=counts.get
    )


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

    complete_text = " ".join(
        item["text"]
        for item in ocr
    )

    normalized = normalize_text(
        complete_text
    )

    for game_map in MAPS:

        if normalize_text(
            game_map
        ) in normalized:

            return game_map

    return None


# ============================================================
# EXTRAER UNA FILA
# ============================================================

def extract_player_row(
    img,
    center_y,
    columns
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

    # Altura de fila proporcional.
    row_height = 24

    y0 = max(
        0,
        int(center_y - row_height)
    )

    y1 = min(
        img.shape[0],
        int(center_y + row_height)
    )

    for column in columns:

        x0 = max(
            0,
            int(column["x0"])
        )

        x1 = min(
            img.shape[1],
            int(column["x1"])
        )

        if x1 <= x0:
            continue

        cell = img[
            y0:y1,
            x0:x1
        ]

        label = normalize_text(
            column["label"]
        )

        numeric = (
            label != "nombre"
            and
            label != "name"
        )

        candidates = read_cell(
            cell,
            numeric=numeric
        )

        # ----------------------------------------------------
        # NOMBRE
        # ----------------------------------------------------

        if (
            "nombre" in label
            or label == "name"
        ):

            result["player"] = choose_player(
                candidates
            )

        # ----------------------------------------------------
        # KILLS
        # ----------------------------------------------------

        elif (
            "elimin" in label
            or label == "kills"
        ):

            result["kills"] = choose_integer(
                candidates,
                0,
                50
            )

        # ----------------------------------------------------
        # ASSISTS
        # ----------------------------------------------------

        elif (
            "asist" in label
            or label == "assists"
        ):

            result["assists"] = choose_integer(
                candidates,
                0,
                50
            )

        # ----------------------------------------------------
        # DAMAGE
        # ----------------------------------------------------

        elif (
            "daño" in label
            or "dano" in label
            or label == "damage"
        ):

            result["damage"] = choose_integer(
                candidates,
                0,
                5000
            )

        # ----------------------------------------------------
        # SURVIVAL
        # ----------------------------------------------------

        elif (
            "superviv" in label
            or "survived" in label
        ):

            result["survival"] = choose_survival(
                candidates
            )

        # ----------------------------------------------------
        # HEALTH
        # ----------------------------------------------------

        elif (
            "vida" in label
            or "health" in label
        ):

            result["hp_recovered"] = choose_integer(
                candidates,
                0,
                5000
            )

        # ----------------------------------------------------
        # RESCUE
        # ----------------------------------------------------

        elif (
            "rescat" in label
            or label == "rescue"
        ):

            result["rescues"] = choose_integer(
                candidates,
                0,
                50
            )

        # ----------------------------------------------------
        # RECALL
        # ----------------------------------------------------

        elif (
            "regresar" in label
            or "recall" in label
        ):

            result["return"] = choose_integer(
                candidates,
                0,
                50
            )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        elif (
            "puntaje" in label
            or label == "rating"
        ):

            result["score"] = choose_score(
                candidates
            )

    # --------------------------------------------------------
    # Si no conseguimos nombre, esta fila no se considera
    # jugador válido.
    # --------------------------------------------------------

    if not result["player"]:
        return None

    return result


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

    original_h, original_w = img.shape[:2]

    # --------------------------------------------------------
    # Para detectar estructura utilizamos una versión
    # reducida si la captura es enorme.
    # --------------------------------------------------------

    detection_img, scale = resize_for_detection(
        img
    )

    # --------------------------------------------------------
    # OCR GENERAL ÚNICO
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        detection_img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=1.4,
        fy=1.4,
        interpolation=cv2.INTER_CUBIC
    )

    detection_ocr = ocr_data(
        gray,
        psm=11
    )

    # Como el OCR se hizo sobre una imagen escalada,
    # convertimos las coordenadas nuevamente.
    ocr = []

    ocr_scale = scale * 1.4

    for item in detection_ocr:

        converted = dict(item)

        converted["x"] = int(
            item["x"] / ocr_scale
        )

        converted["y"] = int(
            item["y"] / ocr_scale
        )

        converted["w"] = int(
            item["w"] / ocr_scale
        )

        converted["h"] = int(
            item["h"] / ocr_scale
        )

        converted["cx"] = int(
            item["cx"] / ocr_scale
        )

        converted["cy"] = int(
            item["cy"] / ocr_scale
        )

        ocr.append(
            converted
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    header = find_header(
        img,
        ocr
    )

    if header is None:

        return {
            "match": match_number,
            "error": (
                "No se pudo localizar "
                "el encabezado de la tabla"
            ),
            "players": []
        }

    # --------------------------------------------------------
    # COLUMNAS
    # --------------------------------------------------------

    columns = find_columns(
        ocr,
        header
    )

    if not columns:

        return {
            "match": match_number,
            "error": (
                "Se encontró el encabezado "
                "pero no se pudieron localizar "
                "las columnas"
            ),
            "players": []
        }

    # --------------------------------------------------------
    # LIMITES DE TABLA
    # --------------------------------------------------------

    bounds = estimate_table_bounds(
        img,
        header,
        columns
    )

    if bounds is None:

        return {
            "match": match_number,
            "error": (
                "No se pudo determinar "
                "la zona de la tabla"
            ),
            "players": []
        }

    left, top, right, bottom = bounds

    # --------------------------------------------------------
    # LÍMITES DE COLUMNAS
    # --------------------------------------------------------

    column_boundaries = build_column_boundaries(
        columns,
        left,
        right
    )

    # --------------------------------------------------------
    # FILAS
    # --------------------------------------------------------

    row_centers = detect_row_centers(
        img,
        bounds,
        ocr,
        header
    )

    players = []

    for center_y in row_centers:

        player = extract_player_row(
            img,
            center_y,
            column_boundaries
        )

        if player is None:
            continue

        # Evitar duplicados.
        duplicate = False

        for existing in players:

            if (
                existing["player"].lower()
                ==
                player["player"].lower()
            ):

                duplicate = True
                break

        if not duplicate:

            player["match"] = match_number

            players.append(
                player
            )

    # --------------------------------------------------------
    # MAPA
    # --------------------------------------------------------

    map_name = detect_map(
        ocr
    )

    # --------------------------------------------------------
    # RESULTADO
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
            "header": header,
            "columns_detected": columns,
            "table_bounds": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
            "row_centers": row_centers,
        }
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


# ============================================================
# ANALIZAR
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
