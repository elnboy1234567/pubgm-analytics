import re
import unicodedata
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from paddleocr import PaddleOCR


app = FastAPI(title="PUBGM Analytics")

# ============================================================
# PADDLEOCR
# ============================================================

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_OCR_SCORE = 0.35

# Palabras que pueden aparecer dentro de la tabla pero
# NO son nombres de jugadores.
BAD_PLAYER_WORDS = {
    "experto",
    "tactico",
    "tactical",
    "expert",
    "wild",
    "shot",
    "cannon",
    "fodder",
    "too",
    "soon",
    "personal",
    "info",
}

HEADER_ALIASES = {
    "player": [
        "nombre",
        "name",
    ],
    "kills": [
        "eliminaciones",
        "eliminations",
        "kills",
    ],
    "assists": [
        "asistencias",
        "assists",
    ],
    "damage": [
        "daño",
        "dano",
        "damage",
    ],
    "survival": [
        "supervivencia",
        "survived",
        "survival",
    ],
    "hp_recovered": [
        "vida",
        "health",
        "restored",
        "recup",
        "recovered",
    ],
    "rescues": [
        "rescates",
        "rescue",
    ],
    "return": [
        "regresar",
        "recall",
    ],
    "score": [
        "puntaje",
        "rating",
        "score",
    ],
}


# ============================================================
# UTILIDADES
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_player_name(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()

    # Quitar espacios internos extra
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Caracteres de borde típicamente introducidos
    # por OCR.
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

    return text.strip()


def is_bad_player_text(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return True

    if len(normalized) < 3:
        return True

    if normalized in BAD_PLAYER_WORDS:
        return True

    # No aceptar solamente números.
    if re.fullmatch(
        r"[\d\s.,:]+",
        normalized
    ):
        return True

    # Textos claramente pertenecientes a la interfaz.
    interface_words = [
        "nombre",
        "name",
        "eliminaciones",
        "eliminations",
        "asistencias",
        "assists",
        "damage",
        "dano",
        "supervivencia",
        "survived",
        "vida",
        "health",
        "rescates",
        "rescue",
        "regresar",
        "recall",
        "puntaje",
        "rating",
        "score",
    ]

    if normalized in interface_words:
        return True

    return False


# ============================================================
# OCR
# ============================================================

def run_ocr(image):
    """
    Ejecuta PaddleOCR y convierte su resultado
    al formato interno que utiliza el proyecto.
    """

    output = []

    try:
        results = ocr.predict(image)

        for result in results:

            # PaddleOCR 3.x expone los resultados
            # mediante result.json
            if hasattr(result, "json"):

                data = result.json

                if callable(data):
                    data = data()

            elif isinstance(result, dict):

                data = result

            else:
                continue

            # Algunas versiones envuelven el resultado
            # dentro de la clave "res".
            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            if not isinstance(data, dict):
                continue

            texts = data.get(
                "rec_texts",
                []
            )

            scores = data.get(
                "rec_scores",
                []
            )

            boxes = data.get(
                "rec_polys",
                data.get(
                    "dt_polys",
                    []
                )
            )

            for box, text, score in zip(
                boxes,
                texts,
                scores
            ):

                add_detection(
                    output,
                    box,
                    text,
                    score
                )

    except Exception as error:

        print(
            f"PaddleOCR error: {error}"
        )

        return []

    return output

    return parse_ocr_result(result)


def parse_ocr_result(result):
    """
    Normaliza diferentes formatos de RapidOCR
    a:

    [
        {
            "text": "...",
            "score": 0.99,
            "box": [[x,y],...],
            "cx": ...,
            "cy": ...,
            "x0": ...,
            "x1": ...,
            "y0": ...,
            "y1": ...
        }
    ]
    """

    output = []

    if result is None:
        return output

    # API nueva: objeto con .boxes .txts .scores
    if hasattr(result, "boxes"):

        boxes = result.boxes
        texts = result.txts
        scores = result.scores

        if boxes is None:
            return output

        for box, text, score in zip(
            boxes,
            texts,
            scores
        ):
            add_detection(
                output,
                box,
                text,
                score
            )

        return output

    # API clásica:
    # [
    #   [box, text, score],
    #   ...
    # ]
    try:
        for item in result:

            if len(item) < 3:
                continue

            box = item[0]
            text = item[1]
            score = item[2]

            add_detection(
                output,
                box,
                text,
                score
            )

    except Exception:
        pass

    return output


def add_detection(
    output,
    box,
    text,
    score
):
    try:

        score = float(score)

        if score < MIN_OCR_SCORE:
            return

        box_array = np.asarray(
            box,
            dtype=float
        )

        if box_array.ndim != 2:
            return

        xs = box_array[:, 0]
        ys = box_array[:, 1]

        x0 = float(np.min(xs))
        x1 = float(np.max(xs))
        y0 = float(np.min(ys))
        y1 = float(np.max(ys))

        output.append(
            {
                "text": str(text).strip(),
                "score": score,
                "box": box_array.tolist(),
                "cx": (x0 + x1) / 2,
                "cy": (y0 + y1) / 2,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "width": x1 - x0,
                "height": y1 - y0,
            }
        )

    except Exception:
        pass


# ============================================================
# IMAGEN
# ============================================================

def decode_image(data: bytes):

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

    return image


def resize_for_ocr(image):

    h, w = image.shape[:2]

    # RapidOCR funciona mejor manteniendo
    # suficiente resolución para textos pequeños.
    #
    # Para capturas de móvil/tablet muy grandes
    # reducimos solamente si es necesario.

    max_dimension = 2200

    largest = max(
        h,
        w
    )

    if largest <= max_dimension:
        return image, 1.0

    scale = (
        max_dimension /
        float(largest)
    )

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# ENCABEZADOS
# ============================================================

def header_matches(text, aliases):

    normalized = normalize_text(text)

    for alias in aliases:

        alias = normalize_text(
            alias
        )

        if alias in normalized:
            return True

    return False


def find_header_detections(
    detections
):

    headers = {}

    for detection in detections:

        text = detection["text"]

        for key, aliases in HEADER_ALIASES.items():

            if header_matches(
                text,
                aliases
            ):

                if key not in headers:
                    headers[key] = []

                headers[key].append(
                    detection
                )

    return headers


def get_header_y(headers):

    ys = []

    for detections in headers.values():

        for detection in detections:

            ys.append(
                detection["cy"]
            )

    if not ys:
        return None

    return float(
        np.median(ys)
    )


# ============================================================
# COLUMNAS
# ============================================================

def build_columns(
    headers,
    image_width
):

    columns = {}

    # Si RapidOCR ha encontrado los encabezados,
    # usamos sus posiciones REALES.
    for key, detections in headers.items():

        if not detections:
            continue

        x_values = [
            d["cx"]
            for d in detections
        ]

        columns[key] = float(
            np.median(x_values)
        )

    # Si algún encabezado no ha sido reconocido,
    # estimamos su posición basándonos en la estructura
    # de PUBG.
    #
    # Esto es solo un fallback.
    if "player" not in columns:
        columns["player"] = image_width * 0.08

    fallback = {
        "kills": 0.36,
        "assists": 0.42,
        "damage": 0.49,
        "survival": 0.56,
        "hp_recovered": 0.64,
        "rescues": 0.73,
        "return": 0.81,
        "score": 0.91,
    }

    for key, ratio in fallback.items():

        if key not in columns:
            columns[key] = (
                image_width * ratio
            )

    return columns


def column_boundaries(
    columns,
    image_width
):

    ordered = sorted(
        columns.items(),
        key=lambda x: x[1]
    )

    boundaries = {}

    for index, (name, center) in enumerate(
        ordered
    ):

        if index == 0:
            left = 0
        else:
            previous_center = (
                ordered[index - 1][1]
            )

            left = (
                previous_center +
                center
            ) / 2

        if index == len(ordered) - 1:
            right = image_width
        else:
            next_center = (
                ordered[index + 1][1]
            )

            right = (
                center +
                next_center
            ) / 2

        boundaries[name] = (
            left,
            right
        )

    return boundaries


def detect_column(
    x,
    boundaries
):

    for name, (
        left,
        right
    ) in boundaries.items():

        if (
            left <= x <= right
        ):
            return name

    return None


# ============================================================
# FILAS
# ============================================================

def group_rows(
    detections,
    header_y,
    image_height
):

    if header_y is None:
        return []

    # Solamente texto debajo de los encabezados.
    candidates = []

    for detection in detections:

        cy = detection["cy"]

        # Evitar el encabezado.
        if cy <= header_y + 15:
            continue

        # Evitar elementos muy inferiores
        # como logos/botones.
        if cy >= image_height * 0.88:
            continue

        candidates.append(
            detection
        )

    if not candidates:
        return []

    # Orden vertical.
    candidates.sort(
        key=lambda d: d["cy"]
    )

    # Agrupación dinámica.
    rows = []

    for detection in candidates:

        added = False

        for row in rows:

            row_center = np.mean(
                [
                    d["cy"]
                    for d in row
                ]
            )

            # Tolerancia proporcional al tamaño
            # del texto.
            tolerance = max(
                18,
                detection["height"] * 1.8
            )

            if abs(
                detection["cy"] -
                row_center
            ) <= tolerance:

                row.append(
                    detection
                )

                added = True
                break

        if not added:

            rows.append(
                [detection]
            )

    # Ordenar filas.
    rows.sort(
        key=lambda row:
        np.mean(
            [d["cy"] for d in row]
        )
    )

    # Fusionar filas que han quedado separadas
    # accidentalmente.
    merged = []

    for row in rows:

        if not merged:

            merged.append(
                row
            )
            continue

        previous = merged[-1]

        previous_center = np.mean(
            [
                d["cy"]
                for d in previous
            ]
        )

        current_center = np.mean(
            [
                d["cy"]
                for d in row
            ]
        )

        if abs(
            current_center -
            previous_center
        ) < 30:

            previous.extend(
                row
            )

        else:

            merged.append(
                row
            )

    return merged


# ============================================================
# NÚMEROS
# ============================================================

def normalize_number_text(
    text
):

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
        "Z": "2",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    return text


def extract_integer(
    texts,
    maximum
):

    candidates = []

    for text in texts:

        text = normalize_number_text(
            text
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

                if (
                    0 <= value <= maximum
                ):
                    candidates.append(
                        value
                    )

            except Exception:
                pass

    if not candidates:
        return None

    # Para una celda debería existir
    # normalmente un único número.
    #
    # Si hay varios, damos prioridad
    # al valor con más apariciones.
    counts = {}

    for value in candidates:

        counts[value] = (
            counts.get(
                value,
                0
            ) + 1
        )

    return max(
        counts,
        key=counts.get
    )


def extract_decimal(
    texts,
    maximum
):

    candidates = []

    for text in texts:

        text = normalize_number_text(
            text
        )

        text = text.replace(
            ",",
            "."
        )

        matches = re.findall(
            r"\d+(?:\.\d+)?",
            text
        )

        for match in matches:

            try:

                value = float(
                    match
                )

                if (
                    0 <= value <= maximum
                ):
                    candidates.append(
                        value
                    )

            except Exception:
                pass

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x:
        candidates.count(x)
    )


# ============================================================
# SUPERVIVENCIA
# ============================================================

def extract_survival(
    texts
):

    for text in texts:

        text = normalize_number_text(
            text
        )

        text = text.replace(
            ",",
            "."
        )

        # 19.3 min
        matches = re.findall(
            r"(\d{1,2})[.:](\d{1,2})",
            text
        )

        if matches:

            minute, decimal = matches[0]

            try:

                return float(
                    f"{minute}.{decimal}"
                )

            except Exception:
                pass

        # 19.3
        matches = re.findall(
            r"\d+\.\d+",
            text
        )

        if matches:

            try:

                value = float(
                    matches[0]
                )

                if 0 < value <= 60:
                    return value

            except Exception:
                pass

    return extract_decimal(
        texts,
        60
    )


# ============================================================
# JUGADOR
# ============================================================

def extract_player(
    detections
):

    candidates = []

    for detection in detections:

        text = clean_player_name(
            detection["text"]
        )

        if is_bad_player_text(
            text
        ):
            continue

        candidates.append(
            detection
        )

    if not candidates:
        return None

    # El nombre está en la parte izquierda
    # de la tabla.
    candidates.sort(
        key=lambda d:
        (
            -d["score"],
            d["cx"]
        )
    )

    # Preferimos texto corto/medio y con
    # confianza alta.
    #
    # Evitamos textos enormes que normalmente
    # son frases de interfaz.
    valid = [
        d for d in candidates
        if 3 <= len(d["text"]) <= 30
    ]

    if not valid:
        valid = candidates

    best = max(
        valid,
        key=lambda d:
        (
            d["score"],
            -abs(
                d["cx"]
            )
        )
    )

    return clean_player_name(
        best["text"]
    )


# ============================================================
# EXTRAER UNA FILA
# ============================================================

def parse_row(
    row,
    boundaries
):

    cells = {
        "player": [],
        "kills": [],
        "assists": [],
        "damage": [],
        "survival": [],
        "hp_recovered": [],
        "rescues": [],
        "return": [],
        "score": [],
    }

    for detection in row:

        column = detect_column(
            detection["cx"],
            boundaries
        )

        if column is None:
            continue

        cells[column].append(
            detection
        )

    player = extract_player(
        cells["player"]
    )

    if not player:
        return None

    result = {
        "player": player,

        "kills": extract_integer(
            [
                d["text"]
                for d in cells["kills"]
            ],
            50
        ),

        "assists": extract_integer(
            [
                d["text"]
                for d in cells["assists"]
            ],
            50
        ),

        "damage": extract_integer(
            [
                d["text"]
                for d in cells["damage"]
            ],
            5000
        ),

        "survival": extract_survival(
            [
                d["text"]
                for d in cells["survival"]
            ]
        ),

        "hp_recovered": extract_integer(
            [
                d["text"]
                for d in cells["hp_recovered"]
            ],
            5000
        ),

        "rescues": extract_integer(
            [
                d["text"]
                for d in cells["rescues"]
            ],
            50
        ),

        "return": extract_integer(
            [
                d["text"]
                for d in cells["return"]
            ],
            50
        ),

        "score": extract_decimal(
            [
                d["text"]
                for d in cells["score"]
            ],
            150
        ),
    }

    return result


# ============================================================
# MAPA
# ============================================================

MAPS = [
    "Erangel",
    "Miramar",
    "Sanhok",
    "Vikendi",
    "Livik",
    "Karakin",
    "Nusa",
    "Rondo",
]


def detect_map(
    detections,
    header_y
):

    # Buscar preferentemente en la parte
    # superior de la captura.
    for detection in detections:

        if (
            header_y is not None
            and
            detection["cy"] > header_y
        ):
            continue

        normalized = normalize_text(
            detection["text"]
        )

        for game_map in MAPS:

            if normalize_text(
                game_map
            ) in normalized:

                return game_map

    return None


# ============================================================
# PLACEMENT
# ============================================================

def detect_placement(
    detections
):

    best = None

    for detection in detections:

        text = normalize_number_text(
            detection["text"]
        )

        match = re.search(
            r"#?\s*(\d{1,3})",
            text
        )

        if not match:
            continue

        value = int(
            match.group(1)
        )

        if not (
            1 <= value <= 100
        ):
            continue

        # El placement suele estar
        # muy arriba y a la izquierda.
        if detection["cx"] > 0.35 * 10000:
            pass

        score = (
            detection["score"]
            +
            (
                1.0 /
                max(
                    detection["cy"],
                    1
                )
            )
        )

        if best is None:
            best = (
                value,
                score
            )

    if best:
        return best[0]

    return None


# ============================================================
# ANALIZAR IMAGEN
# ============================================================

def analyze_image(
    data: bytes,
    match_number: int
):

    original = decode_image(
        data
    )

    original_h, original_w = (
        original.shape[:2]
    )

    image, scale = resize_for_ocr(
        original
    )

    image_h, image_w = (
        image.shape[:2]
    )

    detections = run_ocr(
        image
    )

    if not detections:

        return {
            "match": match_number,
            "map": None,
            "placement": None,
            "players": [],
            "error": "RapidOCR no detectó texto",
        }

    headers = find_header_detections(
        detections
    )

    header_y = get_header_y(
        headers
    )

    columns = build_columns(
        headers,
        image_w
    )

    boundaries = column_boundaries(
        columns,
        image_w
    )

    rows = group_rows(
        detections,
        header_y,
        image_h
    )

    players = []

    for row in rows:

        player = parse_row(
            row,
            boundaries
        )

        if player is None:
            continue

        # Evitar duplicados.
        normalized_name = normalize_text(
            player["player"]
        )

        duplicate = False

        for existing in players:

            if normalize_text(
                existing["player"]
            ) == normalized_name:

                duplicate = True
                break

        if not duplicate:

            player["match"] = match_number

            players.append(
                player
            )

    return {
        "match": match_number,
        "map": detect_map(
            detections,
            header_y
        ),
        "placement": detect_placement(
            detections
        ),
        "players": players,

        # Información de diagnóstico.
        # No afecta a la interfaz actual.
        "debug": {
            "image": {
                "width": original_w,
                "height": original_h,
            },
            "ocr_detections": len(
                detections
            ),
            "header_y": header_y,
            "headers_found": list(
                headers.keys()
            ),
            "columns": columns,
            "rows_detected": len(
                rows
            ),
        },
    }


# ============================================================
# WEB
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    with open(
        "static/index.html",
        encoding="utf-8"
    ) as file:

        return file.read()


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

    for index, file in enumerate(
        files,
        1
    ):

        try:

            data = await file.read()

            result = analyze_image(
                data,
                index
            )

            results.append(
                result
            )

        except Exception as error:

            results.append(
                {
                    "match": index,
                    "error": str(error),
                    "players": [],
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
