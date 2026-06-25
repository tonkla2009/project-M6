"""
Test YOLOv8 Plate Detector — Webcam
====================================
สคริปต์ทดสอบ best.pt อย่างเดียว (ไม่มี OCR / DB / Flask)
ใช้ดูว่าโมเดลจับกรอบป้ายทะเบียนได้แม่นแค่ไหนก่อนเอาไปต่อ EasyOCR

ติดตั้ง:
    pip install ultralytics opencv-python

วิธีใช้:
    1. วาง best.pt ไว้โฟลเดอร์เดียวกับสคริปต์นี้ (หรือแก้ MODEL_PATH ด้านล่าง)
    2. รัน: python test_plate_model.py
    3. กด Q เพื่อออก
"""

import cv2
import re
import os
import time
import difflib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import easyocr

# ============================================================
# ตั้งค่า
# ============================================================
MODEL_PATH   = "best.pt"
FONT_PATH    = "THSarabunNew.ttf"   # ฟอนต์ไทย — ต้องอยู่โฟลเดอร์เดียวกับสคริปต์นี้
CAMERA_INDEX = 2      # ถ้าใช้ OBS Virtual Camera ลองเปลี่ยนเป็น 1 หรือ 2
CONF_THRESH  = 0.4    # ความมั่นใจขั้นต่ำของกล่อง (ลดถ้าหาป้ายไม่เจอ, เพิ่มถ้าจับผิดเยอะ)
OCR_CONF     = 0.3    # ความมั่นใจขั้นต่ำของ EasyOCR
OCR_EVERY_N  = 3       # อ่าน OCR ทุก N frame (ลดภาระ CPU/GPU)
PLATE_SPLIT  = 0.72     # แบ่งป้ายเป็น 2 โซน: บน (เลขทะเบียน) สูง 72% ของป้าย, ล่าง (จังหวัด) 28%

# allowlist เฉพาะตัวอักษร/เลขที่ใช้บนเลขทะเบียนไทย (ลดโอกาส OCR อ่านมั่ว)
PLATE_NUMBER_ALLOWLIST = (
    "0123456789"
    "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
)
COLOR_BOX    = (0, 255, 0)
COLOR_TEXT   = (0, 255, 255)
COLOR_CONF   = (255, 100, 0)

# รายชื่อจังหวัดไทย — ใช้แก้คำที่ OCR อ่านผิด (fuzzy match เอาคำที่ใกล้เคียงที่สุด)
THAI_PROVINCES = [
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
    "กรุงเทพมหานคร",
]

_thai_font = None


def init_thai_font():
    """โหลดฟอนต์ไทยสำหรับวาดข้อความ (cv2.putText วาดภาษาไทยไม่ได้)"""
    global _thai_font
    if not os.path.exists(FONT_PATH):
        print(f"[!] ไม่พบฟอนต์ {FONT_PATH} — ข้อความไทยจะแสดงเป็น ? แทน")
        return
    try:
        _thai_font = ImageFont.truetype(FONT_PATH, 28)
        print(f"[OK] โหลดฟอนต์ไทยเรียบร้อย -> {FONT_PATH}")
    except Exception as e:
        print(f"[!] โหลดฟอนต์ไม่สำเร็จ: {e}")


def draw_thai_text(frame_bgr, text, pos, color_bgr=(0, 255, 255),
                   bg_color=(0, 0, 0), padding=4):
    """วาดข้อความ (รองรับไทย) ลงบนเฟรม BGR ผ่าน PIL"""
    if _thai_font is None:
        cv2.putText(frame_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color_bgr, 2)
        return frame_bgr
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    bbox    = draw.textbbox((0, 0), text, font=_thai_font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y    = pos
    draw.rectangle([x - padding, y - padding, x + tw + padding, y + th + padding],
                   fill=bg_color)
    draw.text((x, y), text, font=_thai_font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def preprocess_roi(roi):
    """เตรียม ROI ก่อนส่ง OCR — ขยายภาพให้ใหญ่พอ + ลด noise + เพิ่มความคมชัด"""
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return roi

    # ขยายให้ความสูงประมาณ 200px ขึ้นไป (ตัวอักษรเล็กเกินไปทำให้ OCR อ่านผิด)
    target_h = 220
    scale    = max(1.0, target_h / roi.shape[0])
    up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # ลด noise แบบรักษาขอบ (ดีกว่า GaussianBlur เฉยๆ สำหรับตัวอักษร)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # เพิ่ม contrast เฉพาะที่ (ช่วยเรื่องแสงไม่สม่ำเสมอ)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # sharpen เบาๆ เพื่อให้เส้นตัวอักษรคมขึ้น
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    _, binary = cv2.threshold(sharpened, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def sort_by_x(ocr_results):
    """เรียงผล OCR จากซ้ายไปขวาตามตำแหน่ง bounding box (กัน OCR คืนผลมาสลับลำดับ)"""
    return sorted(ocr_results, key=lambda r: r[0][0][0])


def read_plate_zones(reader, roi):
    """
    แยกอ่าน ROI ป้ายเป็น 2 โซน:
    - โซนบน  : เลขทะเบียน+พยัญชนะ -> ใช้ allowlist จำกัดตัวอักษร
    - โซนล่าง: ชื่อจังหวัด        -> อ่านแบบเปิดกว้าง แล้วแก้ด้วย correct_province()
    คืนค่า (plate_text, avg_conf)
    """
    h = roi.shape[0]
    split_y = int(h * PLATE_SPLIT)

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

    top_texts = [r[1] for r in top_results if r[2] >= OCR_CONF]
    top_confs = [r[2] for r in top_results if r[2] >= OCR_CONF]
    bot_texts = [r[1] for r in bottom_results if r[2] >= OCR_CONF]
    bot_confs = [r[2] for r in bottom_results if r[2] >= OCR_CONF]

    number_text   = clean_plate_text(top_texts)
    province_text = correct_province(clean_plate_text(bot_texts))

    plate_text = " ".join(t for t in [number_text, province_text] if t)
    all_confs  = top_confs + bot_confs
    avg_conf   = sum(all_confs) / len(all_confs) if all_confs else 0.0

    return plate_text, avg_conf


def correct_province(text):
    """หาคำที่คล้ายชื่อจังหวัดไทยที่สุดในข้อความ แล้วแทนที่คำนั้นถ้าใกล้เคียงพอ"""
    if not text:
        return text
    words = text.split(" ")
    corrected = []
    for w in words:
        if len(w) < 3:           # คำสั้นเกินไป (เลข/ตัวพยัญชนะ) ไม่ต้องเช็คจังหวัด
            corrected.append(w)
            continue
        match = difflib.get_close_matches(w, THAI_PROVINCES, n=1, cutoff=0.5)
        corrected.append(match[0] if match else w)
    return " ".join(corrected)


def clean_plate_text(texts):
    combined = " ".join(texts)
    cleaned  = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9 ]", "", combined)
    return re.sub(r"\s+", " ", cleaned).strip()


def main():
    print("=" * 50)
    print("  ทดสอบ YOLOv8 Plate Detector (best.pt)")
    print("=" * 50)

    print(f"[*] กำลังโหลดโมเดล: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[OK] โหลดโมเดลเรียบร้อย\n")

    init_thai_font()

    print("[*] กำลังโหลด EasyOCR (ครั้งแรกอาจนาน ~1-2 นาที)...")
    reader = easyocr.Reader(["th", "en"], gpu=False)
    print("[OK] โหลด EasyOCR เรียบร้อย\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[!] เปิดกล้อง index {CAMERA_INDEX} ไม่ได้ ลองเปลี่ยนเป็น 0, 1, 2, 3")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[*] เปิดกล้องแล้ว — กด Q เพื่อออก\n")
    fps_time = time.time()
    fps = 0.0
    frame_count = 0
    last_results = []   # (x1,y1,x2,y2, box_conf, text, ocr_conf)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] ไม่ได้รับภาพจากกล้อง")
            time.sleep(0.1)
            continue

        frame_count += 1
        display = frame.copy()

        # ---------- ตรวจจับด้วยโมเดล ----------
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]

        if frame_count % OCR_EVERY_N == 0:
            new_results = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_conf = float(box.conf[0])

                pad = 6
                rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
                rx2, ry2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
                roi = frame[ry1:ry2, rx1:rx2]

                plate_text, avg_conf = read_plate_zones(reader, roi)
                new_results.append((x1, y1, x2, y2, box_conf, plate_text, avg_conf))

                if plate_text:
                    print(f"[OCR] {plate_text}  (box={box_conf:.2f}, ocr={avg_conf:.2f})")

            if new_results:
                last_results = new_results

        # ---------- วาดผล ----------
        for (x1, y1, x2, y2, box_conf, text, ocr_conf) in last_results:
            cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_BOX, 2)

            label = text if text else f"plate {box_conf:.2f}"
            display = draw_thai_text(display, label,
                                     (x1 + 3, max(y1 - 36, 2)),
                                     COLOR_TEXT, (0, 0, 0))

            if text:
                bar_w = int((x2 - x1) * ocr_conf)
                cv2.rectangle(display, (x1, y2 + 2), (x1 + bar_w, y2 + 7), COLOR_CONF, -1)

        # ---------- FPS ----------
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        cv2.putText(display, f"FPS: {fps:.1f}  |  Boxes: {len(last_results)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display, "Plate + OCR Test | Q=quit",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1)

        cv2.imshow("YOLOv8 Plate Detector Test", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] ปิดโปรแกรมเรียบร้อย")


if __name__ == "__main__":
    main()