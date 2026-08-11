import cv2
import pytesseract
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
app = FastAPI(title="PUBGM OCR Diagnostic 2")
TABLE = (0.21, 0.47, 0.80, 0.695)
ROW_Y = [
    0.26,
    0.445,
    0.625,
    0.81,
]
def prepare(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    return gray
def analyze(data):
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
    result = {
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
    for row_number, row_center in enumerate(
        ROW_Y,
        1
    ):
        center = int(
            th * row_center
        )
        # Relatively narrow crop.
        # We want one row without neighboring rows.
        half_height = int(
            th * 0.075
        )
        y0r = max(
            0,
            center - half_height
        )
        y1r = min(
            th,
            center + half_height
        )
        row = table[
            y0r:y1r,
            :
        ]
        gray = prepare(row)
        # ====================================================
        # FULL ROW OCR
        # ====================================================
        data_dict = pytesseract.image_to_data(
            gray,
            config="--psm 6",
            output_type=pytesseract.Output.DICT
        )
        words = []
        total = len(
            data_dict["text"]
        )
        for i in range(total):
            text = str(
                data_dict["text"][i]
            ).strip()
            if not text:
                continue
            try:
                confidence = float(
                    data_dict["conf"][i]
                )
            except Exception:
                confidence = -1
            if confidence < 15:
                continue
            left = int(
                data_dict["left"][i]
            )
            top = int(
                data_dict["top"][i]
            )
            width = int(
                data_dict["width"][i]
            )
            height = int(
                data_dict["height"][i]
            )
            # Convert the OCR coordinates back from
            # the 2x enlarged image to the original row.
            original_x = left / 2
            original_y = top / 2
            original_width = width / 2
            original_height = height / 2
            words.append(
                {
                    "text": text,
                    "confidence": round(
                        confidence,
                        1
                    ),
                    "x": round(
                        original_x,
                        1
                    ),
                    "y": round(
                        original_y,
                        1
                    ),
                    "width": round(
                        original_width,
                        1
                    ),
                    "height": round(
                        original_height,
                        1
                    )
                }
            )
        # Sort from left to right.
        words.sort(
            key=lambda item: (
                item["x"],
                item["y"]
            )
        )
        result["rows"].append(
            {
                "row": row_number,
                "row_center": row_center,
                "crop": {
                    "x": 0,
                    "y": y0r,
                    "width": tw,
                    "height": y1r - y0r
                },
                "words": words
            }
        )
    return result
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
        <meta name="viewport"
              content="width=device-width,
                       initial-scale=1.0">
        <title>PUBGM OCR Diagnostic 2</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 30px auto;
                padding: 20px;
            }
            button {
                padding: 12px 20px;
                font-size: 16px;
            }
            pre {
                background: #111;
                color: #fff;
                padding: 20px;
                border-radius: 8px;
                white-space: pre-wrap;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <h1>PUBGM OCR Diagnostic 2</h1>
        <p>
            Sube UNA captura de resultados.
        </p>
        <input
            type="file"
            id="image"
            accept="image/*"
        >
        <br><br>
        <button onclick="runDiagnostic()">
            Analizar diagnóstico
        </button>
        <h2>Resultado</h2>
        <pre id="result">
Esperando imagen...
        </pre>
        <script>
        async function runDiagnostic() {
            const input =
                document.getElementById("image");
            const output =
                document.getElementById("result");
            if (!input.files.length) {
                output.textContent =
                    "Selecciona una imagen.";
                return;
            }
            output.textContent =
                "Analizando OCR...";
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
                output.textContent =
                    JSON.stringify(
                        data,
                        null,
                        2
                    );
            } catch (error) {
                output.textContent =
                    "ERROR: " + error;
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
        result = analyze(
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
