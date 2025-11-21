from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash, send_file, abort
from ultralytics import YOLO
import cv2
import pytesseract
import threading
import mysql.connector
import datetime
import time
import re
import winsound
import numpy as np
from fuzzywuzzy import fuzz
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os


STREAM_URL_PHONE = "http://192.0.0.4:8080/video"   
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MODEL_PATH = r"E:\Mini Project 5th sem\project\models\license_plate_detector\yolov8n.pt"

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "RANDOM_SECRET_KEY_123_CHANGE_ME")




# -------------------- DATABASE --------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345678",
    database="smart_inspection"
)
cursor = db.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(150) UNIQUE,
    contact VARCHAR(20),
    gender VARCHAR(10),
    identity_card VARCHAR(100),
    password VARCHAR(255),
    created_at DATETIME
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS plates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(50),
    status VARCHAR(20),
    timestamp DATETIME
)
""")
db.commit()

# -------------------- LOAD KNOWN VEHICLES --------------------
cursor.execute("SELECT plate_number FROM known_vehicles")
records = cursor.fetchall()
known_plates = {re.sub(r'[^A-Z0-9]', '', r[0].strip().upper()) for r in records}
print(f"✅ Loaded {len(known_plates)} known vehicle plates from database.")
print(f"📘 Known plates: {known_plates}")

# -------------------- APP STATE --------------------
running = False
paused = False
latest_frame = None
cap = None
current_source = 0       
last_insert_time = 0
lock = threading.Lock()
detection_thread = None

# -------------------- MODEL --------------------
print("⚙️ Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded!")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# -------------------- HELPERS --------------------
def normalize_plate(text):
    if not text:
        return None

    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Indian plate format strict regex:
    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$"

    if re.match(pattern, text):
        return text

    return None

def play_beep():
    try:
        winsound.Beep(1500, 250)
    except Exception:
        pass

def match_with_known_plates(plate, known_plates):
    best_match = None
    best_score = 0
    for known in known_plates:
        score = fuzz.ratio(plate, known)
        if score > best_score:
            best_score = score
            best_match = known
    return best_match if best_score >= 70 else None

def extract_plate_fallback(car_roi):
    try:
        gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [approx], 0, 255, -1)
                x, y = np.where(mask == 255)
                if len(x) == 0:
                    return None
                topx, topy = np.min(x), np.min(y)
                bottomx, bottomy = np.max(x), np.max(y)
                crop = gray[topx:bottomx+1, topy:bottomy+1]
                text = pytesseract.image_to_string(crop, config="--psm 7")
                return normalize_plate(text)
        return None
    except Exception:
        return None

# -------------------- DETECTION LOOP --------------------
def detection_loop():
    global cap, running, paused, latest_frame, last_insert_time, current_source

    stream = current_source
    print(f"🎥 Opening camera source: {stream}")

    cap = cv2.VideoCapture(stream)

    if isinstance(stream, str) and stream.startswith("http"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(1)

    if not cap.isOpened():
        print("❌ Could not open camera!")
        running = False
        return

    while running:
        if paused:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            continue


        if isinstance(stream, str) and stream.startswith("http"):
            frame = cv2.resize(frame, (1280, 720))
            sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, sharpen_kernel)
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            frame = cv2.merge((l, a, b))
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        results = model(frame, verbose=False, conf=0.20)


        results = model(frame, verbose=False)
        annotated = results[0].plot()

        detected_plate = None
        status = "Unknown"

        # YOLO direct plate detection
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if any(k in label for k in ["plate", "license", "number"]):
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                sharp = cv2.filter2D(binary, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
                text = pytesseract.image_to_string(sharp, config='--psm 7 --oem 3')
                plate = normalize_plate(text)
                if plate:
                    matched = match_with_known_plates(plate, known_plates)
                    detected_plate = matched if matched else plate
                    status = "Detected" if matched else "Unknown"
                    color = (0,255,0) if matched else (0,0,255)
                    cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 3)
                    cv2.putText(annotated, f"{detected_plate} ({status})", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    if matched:
                        play_beep()

        # fallback: search car boxes for plate
        if not detected_plate:
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])].lower()
                if label == "car":
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    car_roi = frame[y1:y2, x1:x2]
                    plate = extract_plate_fallback(car_roi)
                    if plate:
                        matched = match_with_known_plates(plate, known_plates)
                        detected_plate = matched if matched else plate
                        status = "Detected" if matched else "Unknown"
                        color = (0,255,0) if matched else (0,0,255)
                        cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 3)
                        cv2.putText(annotated, f"{detected_plate} ({status})", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        if matched:
                            play_beep()
                        break

        if not detected_plate:
            cv2.putText(annotated, "No plate detected...", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255),2)

        now = time.time()
        if detected_plate and (now - last_insert_time >= 5):
            try:
                cursor.execute("""
                               SELECT owner_name, vehicle_color, registration_date 
                               FROM known_vehicles 
                               WHERE plate_number = %s
                               """, (detected_plate,))
                info = cursor.fetchone()
                owner = info[0] if info else None
                color = info[1] if info else None
                reg_date = info[2] if info else None
                cursor.execute("""
                               INSERT INTO plates (plate_number, status, timestamp, owner_name, vehicle_color, registration_date)
                               VALUES (%s, %s, %s, %s, %s, %s)
                               """, (detected_plate, status, datetime.datetime.now(), owner, color, reg_date))
                db.commit()
                last_insert_time = now
                print("DB Updated:", detected_plate)
            except Exception as e:
                print("⚠️ DB Insert Error:", e)


        with lock:
            latest_frame = annotated.copy()
            time.sleep(0.03)


    if cap and cap.isOpened():
        cap.release()
    print("🛑 Detection loop ended.")

# -------------------- AUTH HELPERS --------------------
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user_name=session.get("user_name"))


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    # POST: process form
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    contact = request.form.get("contact", "").strip()
    gender = request.form.get("gender", "").strip()
    identity_card = request.form.get("identity_card", "").strip()
    password = request.form.get("password", "")
    password2 = request.form.get("password2", "")

    if not (name and email and password and password2):
        flash("Please fill required fields.", "error")
        return render_template("register.html")

    if password != password2:
        flash("Passwords do not match.", "error")
        return render_template("register.html")

    hashed = generate_password_hash(password)
    try:
        cursor.execute("""
            INSERT INTO users (name, email, contact, gender, identity_card, password, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (name, email, contact, gender, identity_card, hashed, datetime.datetime.now()))
        db.commit()
    except mysql.connector.IntegrityError:
        flash("Account with this email already exists.", "error")
        return render_template("register.html")
    except Exception as e:
        print("Register error:", e)
        flash("Registration failed. Try again.", "error")
        return render_template("register.html")

    flash("Registration successful. Please login.", "success")
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    # if already logged in redirect to dashboard
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("login.html", error=None)

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    cursor.execute("SELECT id, password, name FROM users WHERE email = %s", (email,))
    row = cursor.fetchone()
    if not row:
        return render_template("login.html", error="Invalid credentials")

    user_id, hashed_pw, name = row
    if not check_password_hash(hashed_pw, password):
        return render_template("login.html", error="Invalid credentials")

    # Successful login
    session["user_id"] = user_id
    session["user_name"] = name
    session["play_sound"] = True
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/user_info")
def user_info():
    if "user_name" not in session:
        return jsonify({"name": None})
    return jsonify({"name": session["user_name"]})


# Camera control endpoints (start/pause/resume/stop/switch)
@app.route('/start')
@login_required
def start():
    global running, detection_thread
    if running:
        return jsonify({"status": "Already running"})
    running = True
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    return jsonify({"status": "Started"})

@app.route('/pause')
@login_required
def pause():
    global paused
    paused = True
    return jsonify({"status": "Paused"})

@app.route('/resume')
@login_required
def resume():
    global paused
    paused = False
    return jsonify({"status": "Resumed"})

@app.route('/stop')
def stop():
    global running, cap, latest_frame
    running = False
    time.sleep(0.5)

    if cap and cap.isOpened():
        cap.release()

    # Clear frame so last image does NOT freeze on screen
    latest_frame = None

    return jsonify({"status": "Stopped. Click Start to begin again"})


@app.route('/switch_camera', methods=['POST'])
@login_required
def switch_camera():
    global current_source, running, cap, detection_thread
    data = request.json or {}
    source = data.get("source", "laptop")
    print("📷 Switching camera to:", source)
    # stop current loop, change source, restart detection thread if running
    running = False
    time.sleep(0.4)
    if cap and cap.isOpened():
        cap.release()
    # set source
    if source == "laptop":
        current_source = 0
    else:
        current_source = STREAM_URL_PHONE
    # restart detection loop if needed
    running = True
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    return jsonify({"status": "source updated"})

@app.route("/refresh_db", methods=["POST"])
def refresh_db():
    global known_plates
    cursor.execute("SELECT plate_number FROM known_vehicles")
    rec = cursor.fetchall()
    known_plates = {re.sub(r'[^A-Z0-9]', '', r[0].strip().upper()) for r in rec}
    return jsonify({"status": "Database reloaded"})


@app.route('/data')
@login_required
def data():
    cursor.execute("""
        SELECT plate_number, status, timestamp, owner_name, vehicle_color, registration_date 
        FROM plates 
        ORDER BY id DESC LIMIT 20
    """)

    rows = cursor.fetchall()
    result = []
    serial = 1

    for r in rows:
        result.append({
            "id": serial,
            "plate": r[0],
            "exists": "Yes" if r[1] == "Detected" else "No",
            "time": r[2].strftime("%Y-%m-%d %H:%M:%S") if r[2] else "",
            "owner_name": r[3] if r[3] else "Unknown",
            "vehicle_color": r[4] if r[4] else "Unknown",
            "registration_date": r[5].strftime("%Y-%m-%d") if r[5] else "Unknown"
        })
        serial += 1

    return jsonify(result)



@app.route("/download_pdf")
def download_pdf():
    # Get detection log
    cursor.execute("SELECT plate_number, status, timestamp FROM plates ORDER BY id DESC")
    logs = cursor.fetchall()

    # Create PDF in memory
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, "Smart Vehicle Inspection — Detailed Vehicle Report")

    p.setFont("Helvetica", 12)
    y -= 40

    if not logs:
        p.drawString(50, y, "No records available.")
    else:
        for log in logs:
            plate = log[0]

            # Fetch known vehicle information
            cursor.execute("""
                SELECT owner_name, vehicle_model, vehicle_color, registration_date, created_at
                FROM known_vehicles 
                WHERE plate_number = %s
            """, (plate,))
            vehicle = cursor.fetchone()

            # Section header
            p.setFont("Helvetica-Bold", 13)
            p.drawString(50, y, f"Plate: {plate}")
            y -= 20

            p.setFont("Helvetica", 11)

            if vehicle:
                owner, model, color, reg_date, created = vehicle

                p.drawString(60, y, f"Owner Name: {owner}")
                y -= 18
                p.drawString(60, y, f"Vehicle Model: {model}")
                y -= 18
                p.drawString(60, y, f"Vehicle Color: {color}")
                y -= 18
                p.drawString(60, y, f"Registration Date: {reg_date}")
                y -= 18
                p.drawString(60, y, f"Added to System: {created}")
                y -= 18
                p.drawString(60, y, f"Last Detected: {log[2]}")
                y -= 25

            else:
                p.drawString(60, y, f"Status: Unknown Vehicle")
                y -= 18
                p.drawString(60, y, f"Last Detected: {log[2]}")
                y -= 25

            # New page if needed
            if y < 80:
                p.showPage()
                y = height - 50

    p.save()
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="vehicle_report.pdf",
        mimetype="application/pdf"
    )


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if latest_frame is None:
                    # send empty frame (black image)
                    empty = np.zeros((480, 640, 3), dtype=np.uint8)
                    ret, buf = cv2.imencode(".jpg", empty)
                else:
                    ret, buf = cv2.imencode(".jpg", latest_frame)

            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')

            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
# -------------------- RUN APP --------------------
if __name__ == "__main__":
    print("🚗 Smart Vehicle License Plate Inspection System (Fuzzy + Fallback + Auth Enabled)")
    app.run(host="0.0.0.0", port=5000, threaded=True)
