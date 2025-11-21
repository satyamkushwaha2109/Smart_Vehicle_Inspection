# 🚗 Smart Vehicle Inspection System

A real-time vehicle monitoring system that automatically **detects vehicles and extracts license plate numbers** using **YOLO (Deep Learning)** and **Tesseract OCR**.  
It provides a **web dashboard built with Flask** and stores vehicle logs in **MySQL** for security and traffic management.

## 📌 Key Features

✔ **Real-time vehicle detection (YOLO)**  
✔ **Automatic License Plate Recognition (OCR)**  
✔ **Works with Webcam & Mobile IP Camera**  
✔ **Flask Dashboard for Live Monitoring**  
✔ **Stores logs into MySQL Database**  
✔ **Duplicate Plate Alerts + Timestamp Logs**  
✔ **Clean UI + Start/Stop Video Control**

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Deep Learning | YOLO |
| OCR | Tesseract / EasyOCR |
| Computer Vision | OpenCV, NumPy |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Database | MySQL |
| Camera Inputs | Webcam / Mobile IP Webcam |
| OS Supported | Windows / Linux |

CREATE DATABASE vehicle_inspection;
CREATE TABLE logs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  plate VARCHAR(50),
  time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  camera_source VARCHAR(50)
);
Run the Application
python main.py

Open in Browser
http://127.0.0.1:5000/

🚦 Real-World Applications

Highway Traffic Monitoring
Automated Toll Collection
Security Gate & Parking Automation
Campus, Society & Industry Surveillance
Crime Investigation (stolen vehicles)
Smart City Analytics

🧬 Future Improvements

🟦 Export reports (CSV/PDF)
🟦 Cloud-based Live Monitoring
🟦 Multi-Camera Support
🟦 Blacklisted Vehicle Alerts
🟦 RTO/Police API Integration

👨‍💻 Developer

Satyam Kushwaha
📍 CSE Engineer
🎯 Passionate in AI, Computer Vision, Security & Web Development




