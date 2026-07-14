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
from flask import (Flask, Response, render_template,
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

    if SHOW_DEBUG:
        show_ocr_input_debug(top_bin, bottom_bin)

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
# Debug windows — ดูภาพที่ผ่านแต่ละขั้นตอนก่อนเข้า OCR
# ──────────────────────────────────────────────
def show_plate_roi_debug(roi):
    """แสดง ROI ดิบที่ได้จาก best.pt (ก่อน preprocess/OCR ใดๆ)"""
    if roi is None or roi.shape[0] == 0 or roi.shape[1] == 0:
        return
    target_w = 320
    scale = target_w / roi.shape[1]
    disp = cv2.resize(roi, (target_w, max(1, int(roi.shape[0]*scale))),
                       interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Debug: Plate ROI (best.pt)", disp)

def show_ocr_input_debug(top_bin, bottom_bin):
    """แสดงภาพ binary ที่ถูกส่งเข้า OCR จริง แยกโซนบน (เลข) / ล่าง (จังหวัด)"""
    def prep(img, w=320):
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return np.zeros((60, w), dtype=np.uint8)
        scale = w / img.shape[1]
        return cv2.resize(img, (w, max(1, int(img.shape[0]*scale))),
                           interpolation=cv2.INTER_NEAREST)
    top_disp    = prep(top_bin)
    bottom_disp = prep(bottom_bin)
    combined = cv2.vconcat([
        top_disp,
        np.full((4, top_disp.shape[1]), 128, dtype=np.uint8),  # เส้นคั่น
        bottom_disp
    ])
    cv2.imshow("Debug: OCR Input (Top=เลขทะเบียน / Bottom=จังหวัด)", combined)

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
                if SHOW_DEBUG:
                    show_plate_roi_debug(roi)   # ภาพดิบจาก best.pt ก่อนเข้า OCR
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
# ──────────────────────────────────────────────
# Routes — Main
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("main.html")

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
    return render_template("dashboard.html", students=students,
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