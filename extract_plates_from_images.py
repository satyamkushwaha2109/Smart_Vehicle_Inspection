import os
import cv2
import pytesseract
import mysql.connector
from ultralytics import YOLO
import datetime
import re
from tqdm import tqdm


DATASET_DIR = r"E:\Mini Project 5th sem\project\models\license_plate_detector\known_dataset\images"
MODEL_PATH = r"E:\Mini Project 5th sem\project\models\license_plate_detector\yolov8n.pt"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ✅ Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Your MySQL PAssword",
    database="smart_inspection"
)
cursor = db.cursor()

# ✅ Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS known_vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) UNIQUE,
    owner_name VARCHAR(100),
    vehicle_model VARCHAR(100),
    vehicle_color VARCHAR(50),
    image_path TEXT,
    registration_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

model = YOLO(MODEL_PATH)


def normalize_plate(s):
    """Clean up the OCR result to valid format like UP70AB1234."""
    s = s.upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    if 6 <= len(s) <= 12:
        return s
    return None


print(f"🚘 Scanning dataset folder: {DATASET_DIR}")

for root, _, files in os.walk(DATASET_DIR):
    for filename in tqdm(files):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(root, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {filename}")
            continue

        # YOLO detection
        results = model(img, verbose=False)
        detections = results[0].boxes

        for box in detections:
            cls_name = model.names[int(box.cls[0])].lower()
            if "plate" not in cls_name and "license" not in cls_name:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(binary, config="--psm 7")
            plate = normalize_plate(text.strip())

            if plate:
                try:
                    cursor.execute("""
                        INSERT IGNORE INTO known_vehicles (plate_number, image_path, registration_date)
                        VALUES (%s, %s, %s)
                    """, (plate, img_path, datetime.date.today()))
                    db.commit()
                    print(f"✅ Stored in DB: {plate}")
                except Exception as e:
                    print(f"⚠️ Error inserting {plate}: {e}")

db.close()
print("✅ All plates from dataset processed and stored successfully!")

