
import re, os, cv2, pytesseract, numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List

app = FastAPI(title="PUBGM Analytics MVP")

# Template coordinates for the PUBG Mobile result screen shown in the supplied samples.
# Coordinates are normalized to the screenshot size.
TABLE = (0.21, 0.47, 0.80, 0.695)
ROW_Y = [0.26, 0.445, 0.625, 0.81]  # relative to table crop; dynamic row detection is attempted first.

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

MAPS = ["Erangel","Miramar","Rondo","Sanhok","Livik","Vikendi","Karakin","Nusa"]

def upscale(im, factor=4):
    return cv2.resize(im, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def clean_name(s):
    s = re.sub(r"[^A-Za-z0-9_*.\-]", "", s)
    # Remove common OCR artefacts at the beginning/end.
    return s.strip("._-")

def ocr_text(cell, psm=7, whitelist=None):
    u = upscale(cell, 4)
    gray = cv2.cvtColor(u, cv2.COLOR_BGR2GRAY)
    cfg=f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    candidates=[]
    for img in (gray, cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]):
        t=pytesseract.image_to_string(img, config=cfg).strip()
        if t: candidates.append(t)
    return candidates

def parse_number(candidates, kind):
    vals=[]
    for t in candidates:
        t=t.replace("O","0").replace("o","0").replace("I","1").replace("l","1").replace(",",".")
        if kind=="survival":
            for m in re.findall(r"\d{1,2}(?:\.\d+)?",t):
                v=float(m)
                if 0 < v <= 60: vals.append(v)
        elif kind=="score":
            for m in re.findall(r"\d+(?:\.\d+)?",t):
                s=m
                if "." not in s and len(s)==3: s=s[:-1]+"."+s[-1]
                try:
                    v=float(s)
                    if 0 <= v <= 150: vals.append(v)
                except: pass
        else:
            for m in re.findall(r"\d+",t):
                vals.append(int(m))
    if not vals: return None
    # Majority vote, then first plausible value.
    counts={}
    for v in vals: counts[v]=counts.get(v,0)+1
    return max(counts, key=lambda v:(counts[v], -vals.index(v)))

def detect_rows(table):
    u=upscale(table,3)
    gray=cv2.cvtColor(u,cv2.COLOR_BGR2GRAY)
    df=pytesseract.image_to_data(gray,config="--psm 6",output_type=pytesseract.Output.DATAFRAME).dropna()
    df["conf"]=pd.to_numeric(df["conf"],errors="coerce")
    df=df[df.conf>=25]
    w=u.shape[1]; h=u.shape[0]
    # Numeric columns contain several words per player row. Cluster their y-centres.
    q=df[(df.left>w*.34) & (df.top>h*.12)].copy()
    ys=sorted((q.top+q.height/2).tolist())
    clusters=[]
    for y in ys:
        if not clusters or y-clusters[-1][-1]>38: clusters.append([y])
        else: clusters[-1].append(y)
    centers=[sum(c)/len(c) for c in clusters if len(c)>=2]
    # Convert from 3x image back to table coordinates.
    centers=[c/3 for c in centers]
    if not centers or len(centers)>8:
        return [int(table.shape[0]*v) for v in ROW_Y]
    # Filter out header-like top cluster.
    centers=[c for c in centers if c>table.shape[0]*.16]
    return centers[:8]

def extract_row(table, yc):
    h,w=table.shape[:2]
    y0=max(0,int(yc-22)); y1=min(h,int(yc+24))
    out={}
    # Name
    a,b=COLS["player"]
    cell=table[y0:y1,int(w*a):int(w*b)]
    names=[]
    for t in ocr_text(cell,7):
        n=clean_name(t)
        if len(n)>=3 and not re.fullmatch(r"\d+",n): names.append(n)
    if not names: return None
    # Prefer the longest plausible token.
    out["player"]=max(names,key=len)
    whitelist="0123456789.,"
    for key,(a,b) in COLS.items():
        if key=="player": continue
        # Slightly specialized vertical crop for survival because the word "min" sits beside the value.
        if key=="survival":
            xa,xb=int(w*.595),int(w*.70)
            cell=table[y0:y1,xa:xb]
            cands=ocr_text(cell,11)
            kind="survival"
        elif key=="score":
            cell=table[y0:y1,int(w*a):int(w*b)]
            cands=ocr_text(cell,7,whitelist)
            kind="score"
        else:
            cell=table[y0:y1,int(w*a):int(w*b)]
            cands=ocr_text(cell,7,whitelist)
            kind="int"
        out[key]=parse_number(cands,kind)
    return out

def parse_metadata(img):
    h,w=img.shape[:2]
    # Map / mode header
    header=img[int(.07*h):int(.47*h),int(.20*w):int(.82*w)]
    txt=pytesseract.image_to_string(header,config="--psm 6")
    mp=None
    low=txt.lower()
    for m in MAPS:
        if m.lower() in low: mp=m; break
    # Placement number is difficult for generic OCR because it is stylized yellow text.
    # We intentionally leave it editable if not confidently recognized.
    placement=None
    return placement, mp

def analyze_image(data, match_number):
    arr=np.frombuffer(data,np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None: raise ValueError("No se pudo leer la imagen")
    h,w=img.shape[:2]
    x0,y0,x1,y1=TABLE
    table=img[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]
    rows=detect_rows(table)
    records=[]
    for yc in rows:
        r=extract_row(table,yc)
        if r and r.get("player"):
            # De-duplicate accidental repeated rows.
            if not any(x["player"].lower()==r["player"].lower() for x in records):
                r["match"]=match_number
                records.append(r)
    placement,map_name=parse_metadata(img)
    return {"match":match_number,"map":map_name,"placement":placement,"players":records}

@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html",encoding="utf-8").read()

@app.post("/api/analyze")
async def analyze(files: List[UploadFile]=File(...)):
    results=[]
    for i,f in enumerate(files,1):
        try:
            results.append(analyze_image(await f.read(),i))
        except Exception as e:
            results.append({"match":i,"error":str(e),"players":[]})
    return JSONResponse({"matches":results})

@app.post("/api/export")
async def export(payload: dict):
    # Export is handled client-side as CSV; this endpoint is reserved for the production backend.
    return {"ok":True}
