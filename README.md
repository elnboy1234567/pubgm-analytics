# PUBG Mobile Analytics — MVP v1.0

## What this version does
- Upload multiple PUBG Mobile result screenshots.
- OCR the result table using OpenCV + Tesseract.
- Detect player rows and extract kills, assists, damage, survival, HP recovered, rescues, return and score.
- Detect map when it appears in the header.
- Consolidate all matches.
- Let the user correct OCR fields before saving.
- Calculate player ranking.
- Generate score/damage charts.
- Export the consolidated dataset as CSV.
- Includes a DEMO button with the three supplied screenshots' verified values.

## Run on Windows/macOS/Linux

Install Tesseract OCR first.
Then:

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000

Open:
http://127.0.0.1:8000

## Docker

docker build -t pubgm-analytics .
docker run --rm -p 8000:8000 pubgm-analytics

## Production roadmap
1. Add Google Vision/Document AI or a multimodal vision model as a second OCR engine.
2. Add confidence scoring and automatic cross-checks.
3. Add support for different PUBG Mobile languages/resolutions/layouts.
4. Store matches in PostgreSQL.
5. Accounts, teams, rosters and historical dashboards.
6. Excel/XLSX + PDF export.
7. Map filters, tournament/scrim tags and player comparison.
8. Mobile app / PWA.
9. Billing and team plans.

The current template is intentionally optimized for the supplied PUBG Mobile result layout; it is a proof-of-concept, not yet a universal PUBG Mobile OCR engine.
