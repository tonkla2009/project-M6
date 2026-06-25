"""
Thai License Plate Reader — Dashboard Edition
===============================================
ฐานข้อมูล : SQLite  (students.db สร้างอัตโนมัติ)
หน้าหลัก  : http://localhost:5000
Dashboard  : http://localhost:5000/dashboard

ติดตั้ง:
    pip install opencv-python easyocr numpy flask pandas Pillow
"""

import cv2, easyocr, numpy as np, pandas as pd
import re, time, threading, os, io, urllib.request, sqlite3, difflib
from PIL import Image, ImageDraw, ImageFont
from flask import (Flask, Response, render_template_string,
                   jsonify, request, redirect, url_for, flash)
from ultralytics import YOLO

# ──────────────────────────────────────────────
# ตั้งค่า
# ──────────────────────────────────────────────
CAMERA_INDEX      = 2
SHOW_DEBUG        = True
CONFIDENCE        = 0.3        # threshold ของ EasyOCR (อ่านตัวอักษร)
PLATE_CONF        = 0.4        # threshold ของ YOLO (หาตำแหน่งป้าย)
FRAME_SKIP        = 2
PLATE_SPLIT       = 0.72       # แบ่งป้ายเป็น 2 โซน: บน (เลขทะเบียน) 72%, ล่าง (จังหวัด) 28%
PLATE_NUMBER_ALLOWLIST = (     # จำกัดตัวอักษรตอนอ่านโซนเลขทะเบียน ลด OCR อ่านมั่ว
    "0123456789"
    "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
)
COLOR_BOX         = (0, 255, 0)
COLOR_BOX_NF      = (0, 140, 255)
COLOR_TEXT        = (0, 255, 255)
COLOR_CONF        = (255, 100, 0)
DB_PATH           = "students.db"
PORT              = 5000
PLATE_MODEL_PATH  = "best.pt"   # โมเดล YOLOv8 ที่เทรนมาจาก Colab

# ──────────────────────────────────────────────
# SQLite helpers
# ──────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT    NOT NULL UNIQUE,
                first_name    TEXT    NOT NULL,
                last_name     TEXT    NOT NULL,
                club          TEXT    DEFAULT '',
                grade         TEXT    NOT NULL,
                room          TEXT    NOT NULL,
                created_at    TEXT    DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.commit()
    print(f"[DB] SQLite พร้อม -> {DB_PATH}")

def db_all_students():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM students ORDER BY grade, room, first_name"
        ).fetchall()
    return [dict(r) for r in rows]

def db_get_student(sid):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE id=?", (sid,)
        ).fetchone()
    return dict(row) if row else None

def db_insert(plate, fname, lname, club, grade, room):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO students
            (license_plate,first_name,last_name,club,grade,room)
            VALUES (?,?,?,?,?,?)
        """, (plate.strip(), fname.strip(), lname.strip(),
              club.strip(), str(grade).strip(), str(room).strip()))
        conn.commit()

def db_update(sid, plate, fname, lname, club, grade, room):
    with get_db() as conn:
        conn.execute("""
            UPDATE students SET license_plate=?,first_name=?,last_name=?,
            club=?,grade=?,room=? WHERE id=?
        """, (plate.strip(), fname.strip(), lname.strip(),
              club.strip(), str(grade).strip(), str(room).strip(), sid))
        conn.commit()

def db_delete(sid):
    with get_db() as conn:
        conn.execute("DELETE FROM students WHERE id=?", (sid,))
        conn.commit()

def db_import_csv(file_bytes):
    """นำเข้าจาก CSV bytes คืนค่า (inserted, skipped, errors)"""
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda col: col.map(
        lambda x: x.strip() if isinstance(x, str) else x))
    inserted = skipped = errors = 0
    with get_db() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO students
                    (license_plate,first_name,last_name,club,grade,room)
                    VALUES (?,?,?,?,?,?)
                """, (row.get("license_plate",""), row.get("first_name",""),
                      row.get("last_name",""),     row.get("club",""),
                      row.get("grade",""),          row.get("room","")))
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                else:
                    skipped  += 1
            except Exception:
                errors += 1
        conn.commit()
    return inserted, skipped, errors

# ──────────────────────────────────────────────
# In-memory DataFrame (reload ทุกครั้งที่ DB เปลี่ยน)
# ──────────────────────────────────────────────
_df_lock = threading.Lock()
_df      = pd.DataFrame()

def reload_df():
    global _df
    rows = db_all_students()
    if rows:
        df = pd.DataFrame(rows)
        df["_key"] = df["license_plate"].apply(normalize)
    else:
        df = pd.DataFrame(columns=[
            "id","license_plate","first_name","last_name",
            "club","grade","room","_key"
        ])
    with _df_lock:
        _df = df
    print(f"[DB] reload DataFrame -> {len(df)} รายการ")

def get_df():
    with _df_lock:
        return _df.copy()

# ──────────────────────────────────────────────
# Thai font (PIL) — แก้ปัญหา ????? บน frame
# ──────────────────────────────────────────────
FONT_PATH = "THSarabunNew.ttf"
FONT_URL  = ("https://github.com/google/fonts/raw/main/ofl/sarabun/"
             "Sarabun-Regular.ttf")
_thai_font_box   = None
_thai_font_small = None

def init_fonts():
    global _thai_font_box, _thai_font_small
    if not os.path.exists(FONT_PATH):
        try:
            print("[*] ดาวน์โหลด Thai font...")
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except Exception as e:
            print(f"[!] ดาวน์โหลด font ไม่ได้: {e}"); return
    try:
        _thai_font_box   = ImageFont.truetype(FONT_PATH, 22)
        _thai_font_small = ImageFont.truetype(FONT_PATH, 16)
        print("[OK] Thai font พร้อม")
    except Exception:
        pass

def draw_thai_text(frame_bgr, text, pos, font,
                   color_bgr=(0,255,255), bg_color=(0,0,0), padding=4):
    if font is None:
        cv2.putText(frame_bgr, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        return frame_bgr
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    bbox    = draw.textbbox((0, 0), text, font=font)
    tw, th  = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x, y    = pos
    draw.rectangle([x-padding, y-padding, x+tw+padding, y+th+padding],
                   fill=bg_color)
    draw.text((x, y), text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ──────────────────────────────────────────────
# OCR + matching helpers
# ──────────────────────────────────────────────
THAI_PROVINCES = {
    "กระบี่","กาญจนบุรี","กาฬสินธุ์","กำแพงเพชร","ขอนแก่น",
    "จันทบุรี","ฉะเชิงเทรา","ชลบุรี","ชัยนาท","ชัยภูมิ","ชุมพร",
    "เชียงราย","เชียงใหม่","ตรัง","ตราด","ตาก","นครนายก","นครปฐม",
    "นครพนม","นครราชสีมา","นครศรีธรรมราช","นครสวรรค์","นนทบุรี",
    "นราธิวาส","น่าน","บึงกาฬ","บุรีรัมย์","ปทุมธานี",
    "ประจวบคีรีขันธ์","ปราจีนบุรี","ปัตตานี","พระนครศรีอยุธยา",
    "พะเยา","พังงา","พัทลุง","พิจิตร","พิษณุโลก","เพชรบุรี",
    "เพชรบูรณ์","แพร่","ภูเก็ต","มหาสารคาม","มุกดาหาร",
    "แม่ฮ่องสอน","ยโสธร","ยะลา","ร้อยเอ็ด","ระนอง","ระยอง",
    "ราชบุรี","ลพบุรี","ลำปาง","ลำพูน","เลย","ศรีสะเกษ","สกลนคร",
    "สงขลา","สตูล","สมุทรปราการ","สมุทรสงคราม","สมุทรสาคร",
    "สระแก้ว","สระบุรี","สิงห์บุรี","สุโขทัย","สุพรรณบุรี",
    "สุราษฎร์ธานี","สุรินทร์","หนองคาย","หนองบัวลำภู","อ่างทอง",
    "อำนาจเจริญ","อุดรธานี","อุตรดิตถ์","อุทัยธานี","อุบลราชธานี",
    "กรุงเทพมหานคร","กทม",
}

def normalize(s):
    return re.sub(r"\s+", "", str(s)).lower()

def extract_plate_only(text):
    m = re.search(r"([ก-ฮ]{1,3})\s*(\d{1,4})", text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    cleaned = text
    for p in THAI_PROVINCES:
        cleaned = cleaned.replace(p, "")
    return cleaned.strip()

def lookup_plate(plate_text):
    df = get_df()
    if df.empty:
        return None
    core = extract_plate_only(plate_text)
    key  = normalize(core)
    # Level 1: exact
    row = df[df["_key"] == key]
    # Level 2: fuzzy (เลขตรง + อักษรต่างกัน <= 1)
    if row.empty:
        nums_ocr = re.sub(r"[^0-9]", "", key)
        th_ocr   = re.sub(r"[^ก-ฮ]",  "", key)
        for _, c in df.iterrows():
            nums_csv = re.sub(r"[^0-9]", "", c["_key"])
            th_csv   = re.sub(r"[^ก-ฮ]",  "", c["_key"])
            if nums_ocr == nums_csv and nums_ocr:
                diff  = sum(a != b for a, b in zip(th_ocr, th_csv))
                diff += abs(len(th_ocr) - len(th_csv))
                if diff <= 1:
                    row = df[df["_key"] == c["_key"]]; break
    # Level 2.5: fuzzy (อักษรตรง/ผิด<=1 + เลขจำนวนหลักเท่ากัน ผิดได้ <=1 หลัก)
    # ช่วยเคส OCR อ่านเลขมั่ว เช่น 7 เป็น 1 (รูปทรงคล้ายกันในบางมุม/แสง)
    if row.empty:
        nums_ocr = re.sub(r"[^0-9]", "", key)
        th_ocr   = re.sub(r"[^ก-ฮ]",  "", key)
        if len(nums_ocr) >= 3:
            for _, c in df.iterrows():
                nums_csv = re.sub(r"[^0-9]", "", c["_key"])
                th_csv   = re.sub(r"[^ก-ฮ]",  "", c["_key"])
                if len(nums_ocr) != len(nums_csv):
                    continue
                num_diff = sum(a != b for a, b in zip(nums_ocr, nums_csv))
                th_diff  = sum(a != b for a, b in zip(th_ocr, th_csv))
                th_diff += abs(len(th_ocr) - len(th_csv))
                if num_diff <= 1 and th_diff <= 1:
                    row = df[df["_key"] == c["_key"]]; break
    # Level 3: number-only fallback
    if row.empty:
        nums_ocr = re.sub(r"[^0-9]", "", key)
        if len(nums_ocr) >= 3:
            for _, c in df.iterrows():
                if re.sub(r"[^0-9]", "", c["_key"]) == nums_ocr:
                    row = df[df["_key"] == c["_key"]]; break
    if row.empty:
        return None
    r = row.iloc[0]
    return {"plate": r["license_plate"],
            "name":  f'{r["first_name"]} {r["last_name"]}',
            "grade": f'ม.{r["grade"]}/{r["room"]}',
            "club":  r.get("club", "-")}

# ──────────────────────────────────────────────
# YOLOv8 plate detector (เทรนมาจาก Colab แทนที่ contour-based เดิม)
# ──────────────────────────────────────────────
_plate_model = None

def init_plate_model():
    global _plate_model
    if not os.path.exists(PLATE_MODEL_PATH):
        print(f"[!] ไม่พบโมเดล {PLATE_MODEL_PATH} — ตรวจสอบว่าไฟล์อยู่โฟลเดอร์เดียวกับ app.py")
        return
    _plate_model = YOLO(PLATE_MODEL_PATH)
    print(f"[OK] โหลดโมเดลตรวจจับป้าย -> {PLATE_MODEL_PATH}")

def find_plate_regions(frame):
    """คืนค่า list ของ (x, y, w, h) จากโมเดล YOLOv8 ที่เทรนไว้"""
    if _plate_model is None:
        return []
    results = _plate_model.predict(frame, conf=PLATE_CONF, verbose=False)[0]
    h_f, w_f = frame.shape[:2]
    regions = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_f, int(x2)), min(h_f, int(y2))
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0:
            regions.append((x1, y1, w, h))
    return regions

def preprocess_roi(roi):
    """เตรียม ROI ก่อนส่ง OCR — ขยายภาพ + ลด noise แบบรักษาขอบ + CLAHE + sharpen + Otsu"""
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return roi
    target_h = 260
    scale    = max(1.0, min(6.0, target_h/roi.shape[0]))
    up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    # sharpen แรงขึ้นเล็กน้อย เพื่อให้เส้นขีดของตัวเลข (เช่นขีดบนของ 7) เด่นชัด
    # ลดโอกาส OCR สับสนระหว่างเลขที่รูปทรงใกล้เคียงกัน เช่น 7 กับ 1
    sharpen_kernel = np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    _, binary = cv2.threshold(sharpened, 0, 255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary

def clean_plate_text(texts):
    combined = " ".join(texts)
    cleaned  = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9 ]", "", combined)
    return re.sub(r"\s+", " ", cleaned).strip()

def sort_by_x(ocr_results):
    """เรียงผล OCR จากซ้ายไปขวาตามตำแหน่ง bounding box (กัน OCR คืนผลมาสลับลำดับ)"""
    return sorted(ocr_results, key=lambda r: r[0][0][0])

def correct_province(text):
    """หาคำที่คล้ายชื่อจังหวัดไทยที่สุดในข้อความ แล้วแทนที่คำนั้นถ้าใกล้เคียงพอ"""
    if not text:
        return text
    words = text.split(" ")
    corrected = []
    for w in words:
        if len(w) < 3:
            corrected.append(w)
            continue
        match = difflib.get_close_matches(w, THAI_PROVINCES, n=1, cutoff=0.5)
        corrected.append(match[0] if match else w)
    return " ".join(corrected)

def read_plate_zones(reader, roi):
    """
    แยกอ่าน ROI ป้ายเป็น 2 โซน:
    - โซนบน  : เลขทะเบียน+พยัญชนะ -> ใช้ allowlist จำกัดตัวอักษร
    - โซนล่าง: ชื่อจังหวัด        -> อ่านแบบเปิดกว้าง แล้วแก้ด้วย correct_province()
    คืนค่า (plate_text, avg_conf)
    """
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return "", 0.0
    h = roi.shape[0]
    split_y = int(h*PLATE_SPLIT)

    top_roi    = roi[0:split_y, :]
    bottom_roi = roi[split_y:h, :]

    top_bin    = preprocess_roi(top_roi)
    bottom_bin = preprocess_roi(bottom_roi)

    top_results = sort_by_x(reader.readtext(
        top_bin, detail=1, paragraph=False,
        allowlist=PLATE_NUMBER_ALLOWLIST
    ))
    bottom_results = sort_by_x(reader.readtext(
        bottom_bin, detail=1, paragraph=False
    ))

    top_texts = [r[1] for r in top_results if r[2] >= CONFIDENCE]
    top_confs = [r[2] for r in top_results if r[2] >= CONFIDENCE]
    bot_texts = [r[1] for r in bottom_results if r[2] >= CONFIDENCE]
    bot_confs = [r[2] for r in bottom_results if r[2] >= CONFIDENCE]

    number_text   = clean_plate_text(top_texts)
    province_text = correct_province(clean_plate_text(bot_texts))

    plate_text = " ".join(t for t in [number_text, province_text] if t)
    all_confs  = top_confs + bot_confs
    avg_conf   = sum(all_confs)/len(all_confs) if all_confs else 0.0

    return plate_text, avg_conf

# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.lock       = threading.Lock()
        self.frame_jpg  = None
        self.last_match = None
        self.last_plate = ""
        self.match_time = 0.0
        self.history    = []

state = AppState()

# ──────────────────────────────────────────────
# Camera thread
# ──────────────────────────────────────────────
def camera_thread(reader):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[!] เปิด camera {CAMERA_INDEX} ไม่ได้"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_count = 0; last_results = []; fps_time = time.time(); fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
        frame_count += 1
        display = frame.copy()

        if frame_count % FRAME_SKIP == 0:
            regions     = find_plate_regions(frame)
            new_results = []
            for (x, y, w, h) in regions:
                pad = 8
                x1,y1 = max(0,x-pad), max(0,y-pad)
                x2,y2 = min(frame.shape[1],x+w+pad), min(frame.shape[0],y+h+pad)
                roi    = frame[y1:y2, x1:x2]
                plate_text, avg_conf = read_plate_zones(reader, roi)
                if plate_text:
                    info  = lookup_plate(plate_text)
                    found = info is not None
                    new_results.append((x1,y1,x2-x1,y2-y1,plate_text,avg_conf,found))
                    with state.lock:
                        state.last_plate = plate_text
                        if found:
                            state.last_match = info
                            state.match_time = time.time()
                            already = any(h["plate"]==info["plate"]
                                          for h in state.history)
                            if not already:
                                state.history.insert(0, {
                                    **info, "time": time.strftime("%H:%M:%S")
                                })
                                state.history = state.history[:10]
                        else:
                            state.last_match = None
            if new_results: last_results = new_results

        for (bx,by,bw,bh,text,conf,found) in last_results:
            color = COLOR_BOX if found else COLOR_BOX_NF
            cv2.rectangle(display,(bx,by),(bx+bw,by+bh),color,2)
            display = draw_thai_text(display, text,
                                     (bx+3, max(by-30,2)),
                                     _thai_font_box, COLOR_TEXT, (0,0,0))
            bar_w = int(bw*conf)
            cv2.rectangle(display,(bx,by+bh+2),(bx+bar_w,by+bh+7),COLOR_CONF,-1)

        now = time.time()
        if now-fps_time >= 1.0:
            fps = FRAME_SKIP/(now-fps_time+1e-9); fps_time = now
        display = draw_thai_text(display, f"FPS:{fps:.1f}",
                                 (10, 8), _thai_font_small, (200,200,200),(0,0,0))
        display = draw_thai_text(display, "Thai Plate Reader | Web UI",
                                 (10, display.shape[0]-24),
                                 _thai_font_small, (150,150,150),(0,0,0))
        if SHOW_DEBUG and frame_count%FRAME_SKIP==0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 50, 150)
            cv2.imshow("Debug: Edges", cv2.resize(edges,(640,360)))
        cv2.waitKey(1)
        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY,80])
        with state.lock:
            state.frame_jpg = buf.tobytes()

    cap.release(); cv2.destroyAllWindows()

# ══════════════════════════════════════════════
# Flask app
# ══════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = "plate_reader_secret_2025"

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
MAIN_HTML = """<!DOCTYPE html><html lang="th"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ระบบตรวจจับป้ายทะเบียน</title>
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--accent:#58a6ff;
--green:#3fb950;--orange:#f0883e;--text:#e6edf3;--muted:#8b949e;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Sarabun',sans-serif;min-height:100vh;}
header{background:var(--surface);border-bottom:1px solid var(--border);
  padding:14px 28px;display:flex;align-items:center;gap:14px;}
.logo{font-size:1.5rem;}
header h1{font-size:1.15rem;font-weight:700;}
header p{font-size:.8rem;color:var(--muted);}
.hright{margin-left:auto;display:flex;align-items:center;gap:10px;}
.badge{background:#1f2a1f;border:1px solid var(--green);color:var(--green);
  padding:4px 12px;border-radius:20px;font-size:.8rem;font-weight:600;}
.badge.offline{background:#2a1f1f;border-color:var(--orange);color:var(--orange);}
.btn-dash{background:var(--accent);color:#0d1117;border:none;padding:7px 16px;
  border-radius:8px;font-size:.85rem;font-weight:700;font-family:'Sarabun',sans-serif;
  cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:5px;}
.btn-dash:hover{opacity:.85;}
.container{max-width:1300px;margin:0 auto;padding:24px 20px;
  display:grid;grid-template-columns:1fr 380px;gap:20px;}
@media(max-width:900px){.container{grid-template-columns:1fr;}}
.cam-wrap{background:#000;border:1px solid var(--border);border-radius:12px;
  overflow:hidden;position:relative;}
.cam-wrap img{width:100%;display:block;}
.cam-label{position:absolute;top:12px;left:12px;background:rgba(0,0,0,.65);
  padding:4px 10px;border-radius:6px;font-size:.75rem;color:var(--muted);}
.sidebar{display:flex;flex-direction:column;gap:16px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;}
.card-title{font-size:.75rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.08em;color:var(--muted);margin-bottom:14px;}
#match-card.found{border-color:var(--green);}
#match-card.missed{border-color:var(--orange);}
.plate-display{font-size:2rem;font-weight:700;letter-spacing:.1em;
  color:var(--accent);margin-bottom:8px;min-height:2.5rem;}
.info-row{display:flex;justify-content:space-between;padding:8px 0;
  border-bottom:1px solid var(--border);font-size:.9rem;}
.info-row:last-child{border-bottom:none;}
.info-label{color:var(--muted);}
.info-val{font-weight:600;}
.status-badge{display:inline-block;padding:3px 10px;border-radius:12px;
  font-size:.75rem;font-weight:600;margin-top:10px;}
.status-badge.found{background:#1a3a1a;color:var(--green);}
.status-badge.missed{background:#3a2010;color:var(--orange);}
.hist-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}
.hist-header .card-title{margin-bottom:0;}
.btn-reset{background:none;border:1px solid var(--border);color:var(--muted);
  padding:3px 10px;border-radius:8px;font-size:.75rem;font-family:'Sarabun',sans-serif;
  cursor:pointer;transition:all .2s;}
.btn-reset:hover{border-color:#e05c5c;color:#e05c5c;background:rgba(224,92,92,.08);}
.hist-item{display:flex;align-items:center;gap:10px;padding:8px 0;
  border-bottom:1px solid var(--border);font-size:.85rem;}
.hist-item:last-child{border-bottom:none;}
.hist-plate{background:#1c2840;color:var(--accent);padding:2px 8px;
  border-radius:6px;font-weight:700;font-size:.8rem;white-space:nowrap;}
.hist-name{flex:1;font-weight:600;}
.hist-grade{color:var(--muted);font-size:.78rem;}
.hist-time{color:var(--muted);font-size:.72rem;white-space:nowrap;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,185,80,.5)}
  70%{box-shadow:0 0 0 10px rgba(63,185,80,0)}100%{box-shadow:0 0 0 0 rgba(63,185,80,0)}}
.pulse{animation:pulse .8s ease-out;}
</style></head><body>
<header>
  <span class="logo">🚗</span>
  <div><h1>ระบบตรวจจับป้ายทะเบียน</h1>
  <p>Thai License Plate Recognition — OBS Virtual Camera</p></div>
  <div class="hright">
    <a href="/dashboard" class="btn-dash">⚙️ Dashboard</a>
    <span class="badge" id="cam-badge">● กล้องออนไลน์</span>
  </div>
</header>
<div class="container">
  <div><div class="cam-wrap">
    <img src="/video_feed" alt="camera">
    <span class="cam-label">OBS Virtual Camera</span>
  </div></div>
  <div class="sidebar">
    <div class="card" id="match-card">
      <div class="card-title">ผลการตรวจสอบ</div>
      <div class="plate-display" id="plate-text">—</div>
      <div id="result-body"><div style="color:var(--muted);font-size:.9rem">รอตรวจจับป้ายทะเบียน...</div></div>
    </div>
    <div class="card">
      <div class="hist-header">
        <div class="card-title">ประวัติล่าสุด</div>
        <button class="btn-reset" onclick="resetHistory()">🗑 ล้างประวัติ</button>
      </div>
      <div id="history-list"><div style="color:var(--muted);font-size:.85rem">ยังไม่มีประวัติ</div></div>
    </div>
  </div>
</div>
<script>
let lastPlate="",lastMatchTime=0;
async function pollStatus(){
  try{
    const d=await fetch("/status").then(r=>r.json());
    document.getElementById("cam-badge").className="badge";
    document.getElementById("cam-badge").textContent="● กล้องออนไลน์";
    const pe=document.getElementById("plate-text");
    if(d.plate!==lastPlate){pe.textContent=d.plate||"—";lastPlate=d.plate;}
    const card=document.getElementById("match-card"),body=document.getElementById("result-body");
    if(d.match&&d.match_time!==lastMatchTime){
      lastMatchTime=d.match_time;const m=d.match;
      card.className="card found pulse";setTimeout(()=>card.classList.remove("pulse"),900);
      body.innerHTML=`<div class="info-row"><span class="info-label">ชื่อ-นามสกุล</span><span class="info-val">${m.name}</span></div>
        <div class="info-row"><span class="info-label">ชั้น / ห้อง</span><span class="info-val">${m.grade}</span></div>
        <div class="info-row"><span class="info-label">สโมสร / ชมรม</span><span class="info-val">${m.club}</span></div>
        <span class="status-badge found">✓ พบข้อมูล</span>`;
    }else if(d.plate&&!d.match){
      card.className="card missed";
      body.innerHTML=`<div style="color:var(--muted);font-size:.9rem;margin-bottom:10px">ไม่พบข้อมูลในระบบ</div>
        <span class="status-badge missed">✗ ไม่พบ</span>`;
    }
    const hel=document.getElementById("history-list");
    if(d.history&&d.history.length>0){
      hel.innerHTML=d.history.map(h=>`<div class="hist-item">
        <span class="hist-plate">${h.plate}</span><span class="hist-name">${h.name}</span>
        <span class="hist-grade">${h.grade}</span><span class="hist-time">${h.time}</span>
      </div>`).join("");
    }else{hel.innerHTML='<div style="color:var(--muted);font-size:.85rem">ยังไม่มีประวัติ</div>';}
  }catch(e){
    document.getElementById("cam-badge").className="badge offline";
    document.getElementById("cam-badge").textContent="● ออฟไลน์";
  }
}
async function resetHistory(){
  await fetch("/reset",{method:"POST"});
  document.getElementById("history-list").innerHTML='<div style="color:var(--muted);font-size:.85rem">ยังไม่มีประวัติ</div>';
}
setInterval(pollStatus,800);pollStatus();
</script></body></html>"""

# ──────────────────────────────────────────────
# DASHBOARD PAGE
# ──────────────────────────────────────────────
DASH_HTML = """<!DOCTYPE html><html lang="th"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard — จัดการนักเรียน</title>
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--accent:#58a6ff;
--green:#3fb950;--orange:#f0883e;--red:#f85149;--text:#e6edf3;--muted:#8b949e;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Sarabun',sans-serif;min-height:100vh;}
header{background:var(--surface);border-bottom:1px solid var(--border);
  padding:14px 28px;display:flex;align-items:center;gap:14px;}
header h1{font-size:1.15rem;font-weight:700;}
header p{font-size:.8rem;color:var(--muted);}
.hright{margin-left:auto;}
a.btn-back{background:none;border:1px solid var(--border);color:var(--text);
  padding:7px 16px;border-radius:8px;font-size:.85rem;font-weight:600;
  font-family:'Sarabun',sans-serif;cursor:pointer;text-decoration:none;
  display:inline-block;}
a.btn-back:hover{border-color:var(--accent);color:var(--accent);}
.btn{display:inline-flex;align-items:center;gap:5px;padding:8px 16px;
  border-radius:8px;font-size:.85rem;font-weight:600;font-family:'Sarabun',sans-serif;
  cursor:pointer;border:none;transition:opacity .2s;text-decoration:none;}
.btn:hover{opacity:.82;}
.btn-success{background:var(--green);color:#0d1117;}
.btn-primary{background:var(--accent);color:#0d1117;}
.btn-danger{background:var(--red);color:#fff;}
.btn-ghost{background:none;border:1px solid var(--border);color:var(--muted);}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent);}
.btn-sm{padding:4px 10px;font-size:.78rem;border-radius:6px;}
.container{max-width:1100px;margin:0 auto;padding:28px 20px;}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px;}
@media(max-width:700px){.stats{grid-template-columns:1fr;}}
.stat-card{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:18px 22px;}
.stat-label{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;}
.stat-val{font-size:2.2rem;font-weight:700;color:var(--accent);margin-top:4px;}
.toolbar{display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap;align-items:center;}
.search-box{flex:1;min-width:200px;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:8px 14px;color:var(--text);font-size:.9rem;
  font-family:'Sarabun',sans-serif;}
.search-box:focus{outline:none;border-color:var(--accent);}
.table-wrap{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;overflow:hidden;}
table{width:100%;border-collapse:collapse;font-size:.9rem;}
thead{background:#1c2128;}
th{padding:11px 16px;text-align:left;font-size:.75rem;text-transform:uppercase;
  letter-spacing:.07em;color:var(--muted);font-weight:600;}
td{padding:11px 16px;border-top:1px solid var(--border);}
tr:hover td{background:rgba(255,255,255,.03);}
.plate-chip{background:#1c2840;color:var(--accent);padding:2px 10px;
  border-radius:6px;font-weight:700;font-size:.85rem;}
.action-btns{display:flex;gap:6px;}
.empty-td{text-align:center;padding:44px;color:var(--muted);font-size:.9rem;}
/* alert */
.alert{padding:10px 14px;border-radius:8px;font-size:.88rem;margin-bottom:18px;}
.alert-success{background:#1a3a1a;border:1px solid var(--green);color:var(--green);}
.alert-error  {background:#3a1010;border:1px solid var(--red);color:var(--red);}
/* modal */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.72);
  z-index:100;align-items:center;justify-content:center;}
.overlay.open{display:flex;}
.modal{background:var(--surface);border:1px solid var(--border);
  border-radius:14px;padding:28px;width:100%;max-width:480px;}
.modal h2{font-size:1.1rem;font-weight:700;margin-bottom:20px;}
.form-group{margin-bottom:14px;}
.form-group label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:5px;}
.form-control{width:100%;background:var(--bg);border:1px solid var(--border);
  border-radius:8px;padding:9px 12px;color:var(--text);font-size:.9rem;
  font-family:'Sarabun',sans-serif;}
.form-control:focus{outline:none;border-color:var(--accent);}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.modal-footer{display:flex;justify-content:flex-end;gap:10px;margin-top:22px;}
/* upload */
.upload-zone{border:2px dashed var(--border);border-radius:10px;padding:28px;
  text-align:center;cursor:pointer;transition:border-color .2s;margin-bottom:4px;}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);}
.upload-zone p{color:var(--muted);font-size:.9rem;margin-top:8px;}
.csv-hint{font-size:.78rem;color:var(--muted);margin-top:8px;line-height:1.6;}
code{background:#1c2128;padding:1px 6px;border-radius:4px;font-size:.82rem;}
</style></head><body>
<header>
  <span style="font-size:1.4rem">⚙️</span>
  <div><h1>Dashboard — จัดการข้อมูลนักเรียน</h1>
  <p>เพิ่ม / แก้ไข / ลบ / อัปโหลด CSV</p></div>
  <div class="hright"><a href="/" class="btn-back">← กลับหน้าหลัก</a></div>
</header>

<div class="container">

  {% with msgs = get_flashed_messages(with_categories=true) %}
  {% if msgs %}{% for cat,msg in msgs %}
  <div class="alert alert-{{cat}}">{{msg}}</div>
  {% endfor %}{% endif %}{% endwith %}

  <!-- Stats -->
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">นักเรียนทั้งหมด</div>
      <div class="stat-val">{{total}}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">จำนวนชั้นเรียน</div>
      <div class="stat-val">{{grades}}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">ป้ายทะเบียนในระบบ</div>
      <div class="stat-val">{{total}}</div>
    </div>
  </div>

  <!-- Toolbar -->
  <div class="toolbar">
    <input class="search-box" id="searchBox"
           placeholder="🔍 ค้นหา ชื่อ / ป้ายทะเบียน / ชั้น / ชมรม..."
           oninput="filterTable()">
    <button class="btn btn-success" onclick="openAdd()">＋ เพิ่มนักเรียน</button>
    <button class="btn btn-primary"
            onclick="document.getElementById('csvOverlay').classList.add('open')">
      ⬆ อัปโหลด CSV
    </button>
  </div>

  <!-- Table -->
  <div class="table-wrap">
    <table id="tbl">
      <thead><tr>
        <th>#</th><th>ป้ายทะเบียน</th><th>ชื่อ-นามสกุล</th>
        <th>ชั้น/ห้อง</th><th>ชมรม</th><th>จัดการ</th>
      </tr></thead>
      <tbody>
      {% for s in students %}
      <tr>
        <td style="color:var(--muted)">{{loop.index}}</td>
        <td><span class="plate-chip">{{s.license_plate}}</span></td>
        <td>{{s.first_name}} {{s.last_name}}</td>
        <td>ม.{{s.grade}}/{{s.room}}</td>
        <td style="color:var(--muted)">{{s.club or '—'}}</td>
        <td><div class="action-btns">
          <button class="btn btn-ghost btn-sm"
            onclick="openEdit({{s.id}},'{{s.license_plate}}','{{s.first_name}}',
            '{{s.last_name}}','{{s.club}}','{{s.grade}}','{{s.room}}')">
            ✏️ แก้ไข</button>
          <button class="btn btn-danger btn-sm"
            onclick="confirmDelete({{s.id}},'{{s.license_plate}}')">
            🗑 ลบ</button>
        </div></td>
      </tr>
      {% else %}
      <tr><td colspan="6" class="empty-td">ยังไม่มีข้อมูลนักเรียน<br>
        กด "เพิ่มนักเรียน" หรืออัปโหลด CSV เพื่อเริ่มต้น</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- Modal: Add / Edit -->
<div class="overlay" id="formOverlay">
<div class="modal">
  <h2 id="modalTitle">เพิ่มนักเรียน</h2>
  <form method="POST" id="studentForm">
    <input type="hidden" name="student_id" id="fId">
    <div class="form-group">
      <label>ป้ายทะเบียน *</label>
      <input class="form-control" name="license_plate" id="fPlate"
             placeholder="เช่น กท 2058" required>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>ชื่อ *</label>
        <input class="form-control" name="first_name" id="fFirst"
               placeholder="ชื่อ" required>
      </div>
      <div class="form-group">
        <label>นามสกุล *</label>
        <input class="form-control" name="last_name" id="fLast"
               placeholder="นามสกุล" required>
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>ชั้น *</label>
        <input class="form-control" name="grade" id="fGrade"
               placeholder="เช่น 6" required>
      </div>
      <div class="form-group">
        <label>ห้อง *</label>
        <input class="form-control" name="room" id="fRoom"
               placeholder="เช่น 1" required>
      </div>
    </div>
    <div class="form-group">
      <label>ชมรม / สโมสร</label>
      <input class="form-control" name="club" id="fClub"
             placeholder="ไม่บังคับ">
    </div>
    <div class="modal-footer">
      <button type="button" class="btn btn-ghost"
              onclick="document.getElementById('formOverlay').classList.remove('open')">
        ยกเลิก</button>
      <button type="submit" class="btn btn-success" id="submitBtn">บันทึก</button>
    </div>
  </form>
</div></div>

<!-- Modal: CSV Upload -->
<div class="overlay" id="csvOverlay">
<div class="modal">
  <h2>อัปโหลด CSV</h2>
  <form method="POST" action="/dashboard/import" enctype="multipart/form-data">
    <div class="upload-zone" id="dropZone"
         onclick="document.getElementById('csvFile').click()"
         ondragover="event.preventDefault();this.classList.add('drag')"
         ondragleave="this.classList.remove('drag')"
         ondrop="handleDrop(event)">
      <div style="font-size:2.5rem">📂</div>
      <p id="dropLabel">คลิกหรือลากไฟล์ CSV มาวางที่นี่</p>
    </div>
    <input type="file" id="csvFile" name="csv_file" accept=".csv"
           style="display:none"
           onchange="document.getElementById('dropLabel').textContent=this.files[0].name">
    <div class="csv-hint">
      รูปแบบคอลัมน์ที่รองรับ:<br>
      <code>license_plate</code>, <code>first_name</code>, <code>last_name</code>,
      <code>club</code>, <code>grade</code>, <code>room</code><br>
      ป้ายทะเบียนซ้ำจะถูกข้ามโดยอัตโนมัติ
    </div>
    <div class="modal-footer">
      <button type="button" class="btn btn-ghost"
              onclick="document.getElementById('csvOverlay').classList.remove('open')">
        ยกเลิก</button>
      <button type="submit" class="btn btn-primary">⬆ นำเข้าข้อมูล</button>
    </div>
  </form>
</div></div>

<!-- Modal: Delete confirm -->
<div class="overlay" id="delOverlay">
<div class="modal" style="max-width:380px">
  <h2>ยืนยันการลบ</h2>
  <p style="margin:16px 0;color:var(--muted)">
    ต้องการลบป้ายทะเบียน
    <strong id="delPlate" style="color:var(--red)"></strong> ใช่ไหม?
  </p>
  <div class="modal-footer">
    <button class="btn btn-ghost"
            onclick="document.getElementById('delOverlay').classList.remove('open')">
      ยกเลิก</button>
    <form method="POST" id="delForm" style="display:inline">
      <button type="submit" class="btn btn-danger">ลบเลย</button>
    </form>
  </div>
</div></div>

<script>
function openAdd(){
  document.getElementById("modalTitle").textContent="เพิ่มนักเรียน";
  document.getElementById("studentForm").action="/dashboard/add";
  document.getElementById("submitBtn").textContent="เพิ่ม";
  ["fId","fPlate","fFirst","fLast","fClub","fGrade","fRoom"]
    .forEach(id=>document.getElementById(id).value="");
  document.getElementById("formOverlay").classList.add("open");
}
function openEdit(id,plate,fname,lname,club,grade,room){
  document.getElementById("modalTitle").textContent="แก้ไขข้อมูล";
  document.getElementById("studentForm").action="/dashboard/edit";
  document.getElementById("submitBtn").textContent="บันทึก";
  document.getElementById("fId").value=id;
  document.getElementById("fPlate").value=plate;
  document.getElementById("fFirst").value=fname;
  document.getElementById("fLast").value=lname;
  document.getElementById("fClub").value=club;
  document.getElementById("fGrade").value=grade;
  document.getElementById("fRoom").value=room;
  document.getElementById("formOverlay").classList.add("open");
}
function confirmDelete(id,plate){
  document.getElementById("delPlate").textContent=plate;
  document.getElementById("delForm").action="/dashboard/delete/"+id;
  document.getElementById("delOverlay").classList.add("open");
}
function handleDrop(e){
  e.preventDefault();
  document.getElementById("dropZone").classList.remove("drag");
  const f=e.dataTransfer.files[0];
  if(f){document.getElementById("csvFile").files=e.dataTransfer.files;
  document.getElementById("dropLabel").textContent=f.name;}
}
function filterTable(){
  const q=document.getElementById("searchBox").value.toLowerCase();
  document.querySelectorAll("#tbl tbody tr").forEach(tr=>{
    tr.style.display=tr.textContent.toLowerCase().includes(q)?"":"none";
  });
}
</script></body></html>"""

# ──────────────────────────────────────────────
# Routes — Main
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(MAIN_HTML)

def gen_frames():
    while True:
        with state.lock: jpg = state.frame_jpg
        if jpg is None: time.sleep(0.05); continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(1/30)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def api_status():
    with state.lock:
        return jsonify({"plate": state.last_plate, "match": state.last_match,
                        "match_time": state.match_time, "history": state.history})

@app.route("/reset", methods=["POST"])
def api_reset():
    with state.lock:
        state.history = []; state.last_match = None; state.last_plate = ""
    return jsonify({"ok": True})

# ──────────────────────────────────────────────
# Routes — Dashboard
# ──────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    students = db_all_students()
    grades   = len(set(s["grade"] for s in students))
    return render_template_string(DASH_HTML, students=students,
                                  total=len(students), grades=grades)

@app.route("/dashboard/add", methods=["POST"])
def dashboard_add():
    try:
        db_insert(request.form["license_plate"],
                  request.form["first_name"], request.form["last_name"],
                  request.form.get("club",""),
                  request.form["grade"], request.form["room"])
        reload_df()
        flash(f"เพิ่ม {request.form['first_name']} {request.form['last_name']} เรียบร้อย ✓",
              "success")
    except sqlite3.IntegrityError:
        flash(f"ป้ายทะเบียน '{request.form['license_plate']}' มีอยู่แล้วในระบบ", "error")
    except Exception as e:
        flash(f"เกิดข้อผิดพลาด: {e}", "error")
    return redirect(url_for("dashboard"))

@app.route("/dashboard/edit", methods=["POST"])
def dashboard_edit():
    try:
        db_update(request.form["student_id"],
                  request.form["license_plate"],
                  request.form["first_name"], request.form["last_name"],
                  request.form.get("club",""),
                  request.form["grade"], request.form["room"])
        reload_df()
        flash("แก้ไขข้อมูลเรียบร้อย ✓", "success")
    except Exception as e:
        flash(f"เกิดข้อผิดพลาด: {e}", "error")
    return redirect(url_for("dashboard"))

@app.route("/dashboard/delete/<int:sid>", methods=["POST"])
def dashboard_delete(sid):
    try:
        s = db_get_student(sid)
        db_delete(sid); reload_df()
        name = f"{s['first_name']} {s['last_name']}" if s else ""
        flash(f"ลบ {name} เรียบร้อย", "success")
    except Exception as e:
        flash(f"เกิดข้อผิดพลาด: {e}", "error")
    return redirect(url_for("dashboard"))

@app.route("/dashboard/import", methods=["POST"])
def dashboard_import():
    f = request.files.get("csv_file")
    if not f or not f.filename.endswith(".csv"):
        flash("กรุณาเลือกไฟล์ .csv", "error")
        return redirect(url_for("dashboard"))
    inserted, skipped, errors = db_import_csv(f.read())
    reload_df()
    flash(f"นำเข้าสำเร็จ {inserted} รายการ  |  ซ้ำ (ข้าม) {skipped}  |  ผิดพลาด {errors}",
          "success")
    return redirect(url_for("dashboard"))

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  Thai Plate Reader — Dashboard Edition")
    print("="*55)
    print(f"  Camera : {CAMERA_INDEX}  |  DB : {DB_PATH}  |  Port : {PORT}")
    init_db()
    reload_df()
    init_fonts()
    init_plate_model()
    print("\n[*] กำลังโหลด EasyOCR (ครั้งแรกอาจนาน ~1-2 นาที)...")
    reader = easyocr.Reader(["th", "en"], gpu=False)
    print("[OK] EasyOCR พร้อม\n")
    t = threading.Thread(target=camera_thread, args=(reader,), daemon=True)
    t.start()
    print(f"[OK] หน้าหลัก  -> http://localhost:{PORT}")
    print(f"[OK] Dashboard -> http://localhost:{PORT}/dashboard")
    print("     กด Ctrl+C เพื่อหยุด\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)