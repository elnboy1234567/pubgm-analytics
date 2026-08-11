import cv2
import pytesseract
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
app = FastAPI(title="PUBGM OCR Diagnostic 3")
TABLE = (0.21, 0.47, 0.80, 0.695)
# Posiciones aproximadas actuales.
ROW_Y = [
    0.26,
    0.445,
    0.625,
    0.81,
]
def prepare(image, threshold=False):
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
    if threshold:
        gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )[1]
    return gray
def run_ocr(image, psm):
    return pytesseract.image_to_string(
        image,
        config=f"--psm {psm}"
    ).strip()
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
    # --------------------------------------------------------
    # Test several vertical positions around each expected row.
    # --------------------------------------------------------
    offsets = [
        -0.045,
        -0.025,
        -0.010,
        0.000,
        0.010,
        0.025,
        0.045
    ]
    for row_number, base in enumerate(
        ROW_Y,
        1
    ):
        tests = []
        for offset in offsets:
            center_relative = (
                base + offset
            )
            center = int(
                th * center_relative
            )
            half_height = int(
                th * 0.055
            )
            ya = max(
                0,
                center - half_height
            )
            yb = min(
                th,
                center + half_height
            )
            # ------------------------------------------------
            # NAME ONLY
            # ------------------------------------------------
            name_cell = table[
                ya:yb,
                0:int(tw * 0.30)
            ]
            normal = prepare(
                name_cell,
                False
            )
            threshold = prepare(
                name_cell,
                True
            )
            normal_psm7 = run_ocr(
                normal,
                7
            )
            normal_psm11 = run_ocr(
                normal,
                11
            )
            threshold_psm7 = run_ocr(
                threshold,
                7
            )
            threshold_psm11 = run_ocr(
                threshold,
                11
            )
            # ------------------------------------------------
            # FULL ROW
            # ------------------------------------------------
            row_cell = table[
                ya:yb,
                :
            ]
            row_normal = prepare(
                row_cell,
                False
            )
            full_psm6 = run_ocr(
                row_normal,
                6
            )
            full_psm11 = run_ocr(
                row_normal,
                11
            )
            tests.append(
                {
                    "offset": offset,
                    "center": round(
                        center_relative,
                        4
                    ),
                    "crop": {
                        "y": ya,
                        "height": yb - ya
                    },
                    "name": {
                        "normal_psm7":
                            normal_psm7,
                        "normal_psm11":
                            normal_psm11,
                        "threshold_psm7":
                            threshold_psm7,
                        "threshold_psm11":
                            threshold_psm11
                    },
                    "full_row": {
                        "psm6":
                            full_psm6,
                        "psm11":
                            full_psm11
                    }
                }
            )
        result["rows"].append(
            {
                "row": row_number,
                "base": base,
                "tests": tests
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
        <title>PUBGM OCR Diagnostic 3</title>
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
        <h1>PUBGM OCR Diagnostic 3</h1>
        <p>
            Sube UNA captura de resultados de PUBG Mobile.
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
