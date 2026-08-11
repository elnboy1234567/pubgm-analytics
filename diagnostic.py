import re
import cv2
import pytesseract
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List

app = FastAPI(title="PUBGM OCR Diagnostic")

# ============================================================
# CURRENT TABLE COORDINATES
# ============================================================

TABLE = (0.21, 0.47, 0.80, 0.695)

ROW_Y = [
    0.26,
    0.445,
    0.625,
    0.81,
]

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


def ocr(cell, psm=7, whitelist=None):

    if cell is None or cell.size == 0:
        return ""

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

    config = f"--psm {psm}"

    if whitelist:
        config += (
            f" -c tessedit_char_whitelist={whitelist}"
        )

    return pytesseract.image_to_string(
        gray,
        config=config
    ).strip()


def analyze_diagnostic(data):

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

    th, tw = table.shape[:2]

    diagnostic = {
        "image_size": {
            "width": w,
            "height": h
        },
        "table_size": {
            "width": tw,
            "height": th
        },
        "rows": []
    }

    # ========================================================
    # ANALYZE EACH POSSIBLE PLAYER ROW
    # ========================================================

    for row_number, row_center in enumerate(
        ROW_Y,
        1
    ):

        # Wider crop for diagnosis so we can see exactly
        # what OCR is reading.
        half_height = int(
            th * 0.10
        )

        center = int(
            th * row_center
        )

        y0r = max(
            0,
            center - half_height
        )

        y1r = min(
            th,
            center + half_height
        )

        row_result = {
            "row": row_number,
            "row_center": row_center,
            "raw": {}
        }

        # ====================================================
        # EACH COLUMN
        # ====================================================

        for key, (a, b) in COLS.items():

            xa = int(
                tw * a
            )

            xb = int(
                tw * b
            )

            cell = table[
                y0r:y1r,
                xa:xb
            ]

            if key == "player":

                text = ocr(
                    cell,
                    psm=7
                )

            elif key == "survival":

                # Wider diagnostic crop.
                xa2 = int(
                    tw * 0.58
                )

                xb2 = int(
                    tw * 0.72
                )

                cell = table[
                    y0r:y1r,
                    xa2:xb2
                ]

                text = ocr(
                    cell,
                    psm=7,
                    whitelist="0123456789.,"
                )

            else:

                text = ocr(
                    cell,
                    psm=7,
                    whitelist="0123456789.,"
                )

            row_result["raw"][key] = text

        diagnostic["rows"].append(
            row_result
        )

    return diagnostic


@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>PUBGM OCR Diagnostic</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
            }

            button {
                padding: 12px 20px;
                font-size: 16px;
                cursor: pointer;
            }

            pre {
                background: #111;
                color: #fff;
                padding: 20px;
                overflow-x: auto;
                white-space: pre-wrap;
            }
        </style>
    </head>

    <body>

        <h1>PUBGM OCR Diagnostic</h1>

        <p>
            Sube UNA captura de resultados de PUBG Mobile.
        </p>

        <input
            type="file"
            id="image"
            accept="image/*"
        >

        <br><br>

        <button onclick="analyze()">
            Analizar diagnóstico
        </button>

        <h2>Resultado OCR</h2>

        <pre id="result">
Esperando imagen...
        </pre>

        <script>

        async function analyze() {

            const input =
                document.getElementById("image");

            const result =
                document.getElementById("result");

            if (!input.files.length) {

                result.textContent =
                    "Selecciona una imagen.";

                return;
            }

            result.textContent =
                "Analizando...";

            const form =
                new FormData();

            form.append(
                "file",
                input.files[0]
            );

            try {

                const response =
                    await fetch(
                        "/api/diagnostic",
                        {
                            method: "POST",
                            body: form
                        }
                    );

                const data =
                    await response.json();

                result.textContent =
                    JSON.stringify(
                        data,
                        null,
                        2
                    );

            } catch (error) {

                result.textContent =
                    "ERROR: " +
                    error;

            }

        }

        </script>

    </body>
    </html>
    """


@app.post("/api/diagnostic")
async def diagnostic(
    file: UploadFile = File(...)
):

    try:

        data = await file.read()

        result = analyze_diagnostic(
            data
        )

        return JSONResponse(
            result
        )

    except Exception as error:

        return JSONResponse(
            {
                "error": str(error)
            },
            status_code=500
        )
