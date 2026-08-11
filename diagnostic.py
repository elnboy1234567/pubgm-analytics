import cv2
import pytesseract
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
app = FastAPI(title="PUBGM OCR Fast Diagnostic")
TABLE = (0.21, 0.47, 0.80, 0.695)
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
    # Una sola ampliación.
    gray = cv2.cvtColor(
        table,
        cv2.COLOR_BGR2GRAY
    )
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    # Una sola lectura OCR.
    data_ocr = pytesseract.image_to_data(
        gray,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )
    words = []
    total = len(
        data_ocr["text"]
    )
    for i in range(total):
        text = str(
            data_ocr["text"][i]
        ).strip()
        if not text:
            continue
        try:
            confidence = float(
                data_ocr["conf"][i]
            )
        except Exception:
            confidence = -1
        if confidence < 15:
            continue
        left = int(
            data_ocr["left"][i]
        )
        top = int(
            data_ocr["top"][i]
        )
        width = int(
            data_ocr["width"][i]
        )
        height = int(
            data_ocr["height"][i]
        )
        # Volvemos a las coordenadas de la tabla original.
        x = left / 2
        y = top / 2
        word_width = width / 2
        word_height = height / 2
        center_y = (
            y + word_height / 2
        )
        words.append(
            {
                "text": text,
                "confidence": round(
                    confidence,
                    1
                ),
                "x": round(
                    x,
                    1
                ),
                "y": round(
                    y,
                    1
                ),
                "center_y": round(
                    center_y,
                    1
                ),
                "width": round(
                    word_width,
                    1
                ),
                "height": round(
                    word_height,
                    1
                )
            }
        )
    # Orden vertical y después horizontal.
    words.sort(
        key=lambda item: (
            item["center_y"],
            item["x"]
        )
    )
    # ========================================================
    # AGRUPACIÓN AUTOMÁTICA POR FILA
    # ========================================================
    rows = []
    for word in words:
        cy = word["center_y"]
        assigned = False
        for row in rows:
            if abs(
                cy - row["center_y"]
            ) <= 18:
                row["words"].append(
                    word
                )
                # Actualizar centro.
                ys = [
                    x["center_y"]
                    for x in row["words"]
                ]
                row["center_y"] = (
                    sum(ys) / len(ys)
                )
                assigned = True
                break
        if not assigned:
            rows.append(
                {
                    "center_y": cy,
                    "words": [word]
                }
            )
    rows.sort(
        key=lambda row:
        row["center_y"]
    )
    # Limpiar filas que parezcan cabeceras/bordes.
    useful_rows = []
    for row in rows:
        useful_words = []
        for word in row["words"]:
            text = word["text"]
            # Ignorar únicamente símbolos de borde.
            if text in ["|", "—", "_"]:
                continue
            useful_words.append(
                word
            )
        if useful_words:
            row["words"] = sorted(
                useful_words,
                key=lambda item:
                item["x"]
            )
            useful_rows.append(
                row
            )
    result_rows = []
    for number, row in enumerate(
        useful_rows,
        1
    ):
        result_rows.append(
            {
                "row": number,
                "center_y": round(
                    row["center_y"],
                    1
                ),
                "words": row["words"]
            }
        )
    return {
        "image_size": {
            "width": w,
            "height": h
        },
        "table_size": {
            "width": tw,
            "height": th
        },
        "detected_words": len(words),
        "detected_rows": len(
            result_rows
        ),
        "rows": result_rows
    }
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
        <title>PUBGM OCR Fast Diagnostic</title>
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
                color: white;
                padding: 20px;
                border-radius: 8px;
                white-space: pre-wrap;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <h1>PUBGM OCR Fast Diagnostic</h1>
        <p>
            Sube UNA captura de resultados.
        </p>
        <input
            type="file"
            id="image"
            accept="image/*"
        >
        <br><br>
        <button onclick="analyze()">
            Analizar
        </button>
        <h2>Resultado</h2>
        <pre id="result">
Esperando imagen...
        </pre>
        <script>
        async function analyze() {
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
