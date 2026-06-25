#"""
Thai License Plate Detector & Reader
====================================
ใช้ OBS Virtual Camera เป็น input
- ตรวจจับป้ายทะเบียนด้วย OpenCV (contour + color filtering)
- อ่านตัวอักษรด้วย EasyOCR (รองรับภาษาไทย + อังกฤษ)

ติดตั้ง dependencies:
    pip install opencv-python easyocr numpy

หรือถ้าใช้ GPU (เร็วกว่า):
    pip install opencv-python easyocr numpy torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""

import cv2
import easyocr
import numpy as np
import time
import re

# ============================================================
# ตั้งค่าทั่วไป
# ============================================================
CAMERA_INDEX = 2       # OBS Virtual Camera มักเป็น index 1 หรือ 2
                        # ถ้าไม่ขึ้นให้ลอง 0, 1, 2, 3 ตามลำดับ
SHOW_DEBUG   = True     # แสดง debug window (binary + contour)
CONFIDENCE   = 0.3      # threshold ความมั่นใจของ OCR (0.0 - 1.0)
FRAME_SKIP   = 2        # อ่าน OCR ทุก N frame (ลด CPU/GPU)

# สีกรอบ (BGR)
COLOR_BOX    = (0, 255, 0)      # กรอบป้าย
COLOR_TEXT   = (0, 255, 255)    # ข้อความบน frame
COLOR_CONF   = (255, 100, 0)    # confidence bar


# ============================================================
# ฟังก์ชันค้นหาป้ายทะเบียน
# ============================================================
def find_plate_regions(frame):
    """
    คืนค่า list ของ (x, y, w, h) บริเวณที่น่าจะเป็นป้ายทะเบียน
    ใช้ edge detection + contour filtering
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 50, 150)

    # ขยาย edge เพื่อเชื่อมเส้น
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_frame, w_frame = frame.shape[:2]
    regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area        = w * h
        aspect      = w / h if h > 0 else 0

        # กรอง: อัตราส่วน ~2:1 ถึง 5:1, ขนาดพอสมควร
        if (1.8 < aspect < 6.0
                and area > 2000
                and w > 60
                and h > 20
                and w < w_frame * 0.8):
            regions.append((x, y, w, h))

    # รวม region ที่ซ้อนทับกัน
    regions = merge_overlapping(regions)
    return regions


def merge_overlapping(boxes, overlap_thresh=0.3):
    """รวม bounding box ที่ซ้อนกันเกิน threshold"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    merged = []
    used   = [False] * len(boxes)

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if used[i]:
            continue
        for j, (x2, y2, w2, h2) in enumerate(boxes[i+1:], i+1):
            if used[j]:
                continue
            # คำนวณ intersection
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            inter = ix * iy
            union = w1*h1 + w2*h2 - inter
            iou   = inter / union if union > 0 else 0
            if iou > overlap_thresh:
                used[j] = True
        merged.append(boxes[i])

    return merged


def preprocess_roi(roi):
    """เตรียม ROI ก่อนส่ง OCR"""
    # ขยายขนาด
    scale  = max(1, 200 // roi.shape[0])
    roi_up = cv2.resize(roi, None, fx=scale*1.5, fy=scale*1.5,
                        interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)

    # ปรับ contrast
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray   = clahe.apply(gray)

    # threshold
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def clean_plate_text(texts):
    """
    กรองและทำความสะอาดข้อความจาก OCR
    คืนค่า string รวม
    """
    combined = " ".join(texts)
    # ลบอักขระที่ไม่ใช่ ไทย/อังกฤษ/ตัวเลข/ช่องว่าง
    cleaned  = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9 ]", "", combined)
    cleaned  = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 55)
    print("  Thai License Plate Reader — OBS Virtual Camera")
    print("=" * 55)
    print(f"  กำลังเปิด camera index: {CAMERA_INDEX}")
    print("  กด  Q  เพื่อออก")
    print("=" * 55)

    # โหลด EasyOCR (ครั้งแรกจะโหลด model ~500MB)
    print("\n[*] กำลังโหลด EasyOCR model (ภาษาไทย + อังกฤษ)...")
    reader = easyocr.Reader(["th", "en"], gpu=False)
    print("[✓] โหลด EasyOCR เสร็จแล้ว\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[!] เปิด camera {CAMERA_INDEX} ไม่ได้")
        print("    ลองเปลี่ยน CAMERA_INDEX เป็น 0, 1, 2 หรือ 3")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count  = 0
    last_results = []   # (x,y,w,h, text, conf)
    fps_time     = time.time()
    fps          = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] ไม่ได้รับ frame — ตรวจสอบ OBS Virtual Camera")
            time.sleep(0.1)
            continue

        frame_count += 1
        display = frame.copy()

        # ---------- ค้นหาป้าย & OCR (ทุก FRAME_SKIP frame) ----------
        if frame_count % FRAME_SKIP == 0:
            regions = find_plate_regions(frame)
            new_results = []

            for (x, y, w, h) in regions:
                # เพิ่ม padding
                pad = 8
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                roi    = frame[y1:y2, x1:x2]
                binary = preprocess_roi(roi)

                # OCR บน binary image
                results = reader.readtext(binary, detail=1,
                                          paragraph=False,
                                          allowlist=None)

                texts = [r[1] for r in results if r[2] >= CONFIDENCE]
                confs = [r[2] for r in results if r[2] >= CONFIDENCE]
                avg_conf = sum(confs) / len(confs) if confs else 0.0

                plate_text = clean_plate_text(texts)
                if plate_text:
                    new_results.append((x1, y1, x2-x1, y2-y1,
                                        plate_text, avg_conf))
                    print(f"[OCR] {plate_text}  (conf={avg_conf:.2f})")

            if new_results:
                last_results = new_results

        # ---------- วาด bounding box + ข้อความ ----------
        for (bx, by, bw, bh, text, conf) in last_results:
            # กรอบ
            cv2.rectangle(display,
                          (bx, by), (bx+bw, by+bh),
                          COLOR_BOX, 2)

            # พื้นหลังข้อความ
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display,
                          (bx, by - th - 10),
                          (bx + tw + 6, by),
                          (0, 0, 0), -1)

            # ข้อความ
            cv2.putText(display, text,
                        (bx + 3, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, COLOR_TEXT, 2)

            # confidence bar
            bar_w = int(bw * conf)
            cv2.rectangle(display,
                          (bx, by + bh + 2),
                          (bx + bar_w, by + bh + 7),
                          COLOR_CONF, -1)

        # ---------- FPS ----------
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time) if (now - fps_time) > 0 else 0
            # reset every second
            fps      = FRAME_SKIP / (now - fps_time + 1e-9)
            fps_time = now

        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)
        cv2.putText(display, "Thai Plate Reader | Q=quit",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1)

        cv2.imshow("Thai License Plate Reader", display)

        # ---------- debug window ----------
        if SHOW_DEBUG and frame_count % FRAME_SKIP == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur  = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            debug = cv2.resize(edges, (640, 360))
            cv2.imshow("Debug: Edges", debug)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] ปิดโปรแกรมเรียบร้อย")


if __name__ == "__main__":
    main()