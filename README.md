# 🚦 AI Traffic Violation Detector & Dashboard

<div align="center">

# 🤖 Smart AI-Based Traffic Monitoring & Violation Detection System

An AI-powered computer vision system that automatically detects traffic rule violations from CCTV or recorded video footage and displays them on an interactive dashboard for traffic authorities.

![Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/ComputerVision-OpenCV-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-orange?style=for-the-badge)
![DeepSort](https://img.shields.io/badge/Tracking-DeepSort-purple?style=for-the-badge)

</div>

---

# ✨ Features

## 🚨 AI-Based Traffic Violation Detection

- ✅ Signal jumping detection
- ✅ Helmetless riding detection
- ✅ Overspeeding detection
- ✅ Wrong lane driving detection
- ✅ Triple riding detection

---

## 🧠 AI & Computer Vision Features

- ✅ Real-time object detection
- ✅ Vehicle tracking using DeepSort
- ✅ License plate extraction using OCR
- ✅ Automatic violation logging
- ✅ Evidence image capture
- ✅ Timestamp-based monitoring

---

## 📊 Dashboard Features

- ✅ Interactive Streamlit dashboard
- ✅ Dark-themed UI
- ✅ Violation filtering system
- ✅ Officer authentication
- ✅ Real-time updates
- ✅ Violation evidence viewer
- ✅ AI chatbot for querying records

---

# 📌 Project Overview

The **AI Traffic Violation Detector & Dashboard** is designed to reduce manual traffic monitoring using advanced **deep learning and computer vision technologies**.

The system automatically:

- Detects vehicles and riders
- Tracks vehicle movement
- Identifies traffic rule violations
- Extracts license plate numbers
- Stores violation evidence
- Displays incidents in a centralized dashboard

This project supports:

- 🚦 Smart Traffic Monitoring
- 🏙️ Smart City Infrastructure
- 👮 Efficient Law Enforcement
- 🛣️ Road Safety Improvement

---

# ❗ Problem Statement

Traditional traffic monitoring relies heavily on:

- Manual observation by traffic police
- CCTV cameras without intelligence
- Human intervention

This leads to:

- ❌ Missed violations
- ❌ Human errors
- ❌ Delayed enforcement
- ❌ Increased manpower dependency

---

# ✅ Proposed Solution

Our AI system automatically:

- Detects traffic participants
- Tracks vehicle movement
- Identifies violations in real-time
- Extracts license plates using OCR
- Captures visual evidence
- Logs all incidents into a smart dashboard

---

# 🚨 Detected Traffic Violations

| Violation | Description |
|-----------|-------------|
| 🚦 Signal Jumping | Crossing during red light |
| 🪖 Helmetless Riding | Rider without helmet |
| 🚗 Overspeeding | Vehicle exceeding speed limit |
| 🔁 Wrong Lane Driving | Illegal lane movement |
| 👨‍👩‍👦 Triple Riding | More than 2 riders on bike |

---

# 🧠 System Architecture

## 🔄 High-Level Workflow

```text
CCTV / Video Input
        ↓
Frame Extraction using OpenCV
        ↓
Object Detection using YOLOv8
        ↓
Vehicle Tracking using DeepSort
        ↓
License Plate OCR using EasyOCR
        ↓
Violation Detection Logic
        ↓
Evidence Logging (JSON + Images)
        ↓
Dashboard Visualization using Streamlit
```

---

# 🏗️ Project Architecture

| Module | Purpose |
|--------|---------|
| Video Processing | Frame extraction |
| Object Detection | Vehicle & rider detection |
| Tracking System | Multi-object tracking |
| OCR Engine | License plate extraction |
| Violation Engine | Rule violation analysis |
| Dashboard | Monitoring & analytics |

---

# 🛠️ Technology Stack

| Component | Technology |
|--------|-----------|
| Object Detection | YOLOv8 |
| Object Tracking | DeepSort |
| OCR | EasyOCR |
| Video Processing | OpenCV |
| Dashboard | Streamlit |
| Data Handling | Pandas, JSON |
| Language | Python |

---

# 📂 Project Structure

```bash
AI_Traffic_Violation_Detector/
│
├── app/
│   ├── main.py
│   ├── detector.py
│   ├── tracker.py
│   ├── ocr.py
│   ├── violation_engine.py
│   └── dashboard.py
│
├── models/
│   ├── yolov8.pt
│   └── deepsort/
│
├── violations/
│   ├── images/
│   └── logs/
│
├── dashboard/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation & Setup

## 📌 Requirements

Install the following software:

- Python 3.10+
- pip
- VS Code
- Git

---

# 🚀 Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ai-traffic-violation-detector.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd ai-traffic-violation-detector
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Detection System

```bash
python app/main.py
```

---

# 📊 Run Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

# 📡 System Workflow

## 🚦 Traffic Monitoring Process

1️⃣ Upload traffic video or connect CCTV feed  
2️⃣ Frames are processed using OpenCV  
3️⃣ YOLOv8 detects vehicles and riders  
4️⃣ DeepSort tracks vehicle movement  
5️⃣ OCR extracts license plate numbers  
6️⃣ Violation logic identifies offenses  
7️⃣ Evidence images are stored  
8️⃣ Dashboard updates automatically  

---

# 📊 Dashboard Features

## 🌙 Smart Dashboard UI

- Dark mode interface
- Real-time updates
- Officer login system
- Violation analytics
- Search & filtering options

---

## 📸 Evidence Management

Each violation stores:

- Vehicle image
- Timestamp
- License plate number
- Violation type
- Confidence score

---

## 🤖 AI Chatbot Integration

Users can ask queries like:

```text
"Show helmet violations"
"Display red car violations"
"Show overspeeding vehicles"
```

---

# 🧪 Testing

## ✅ Tested Functionalities

- Vehicle detection
- Object tracking
- OCR extraction
- Violation logging
- Dashboard updates
- Real-time filtering

---

## 🛠️ Tools Used

- OpenCV
- Streamlit
- Postman
- Manual testing
- Browser testing

---

# 📈 Impact & Use Cases

## 🚓 Benefits

- Reduces manual monitoring workload
- Enables evidence-based traffic enforcement
- Improves road safety
- Supports automated surveillance systems

---

## 🏙️ Suitable For

- Smart Cities
- Traffic Command Centers
- Highway Monitoring Systems
- Toll Booth Monitoring
- Urban Surveillance Systems

---

# 🔮 Future Enhancements

- 🔴 Live CCTV integration
- 💳 Automatic e-challan generation
- ☁️ Cloud-based dashboard
- 🇮🇳 Improved Indian number plate OCR
- 📱 Mobile application support
- 📊 AI analytics dashboard
- 🛰️ Multi-camera synchronization

---

# 🔐 Security Features

The project includes:

- Secure officer login
- Authentication system
- Protected dashboard access
- Secure evidence storage
- Data logging and tracking

---

# 📸 Screenshots

## 🚦 Dashboard Homepage
(Add dashboard screenshot here)

---

## 🚗 Violation Detection
(Add detection screenshot here)

---

## 📸 License Plate OCR
(Add OCR screenshot here)

---

# 🧾 One-Line Summary

> **A smart AI assistant for traffic monitoring that detects violations in real-time, logs evidence automatically, and simplifies enforcement through an interactive dashboard.**

---

# 👨‍💻 Team & Contributions

This project was developed as part of a:

- Hackathon project
- Academic innovation initiative
- AI-driven public safety solution

---

# 📜 License

This project is intended for:

- Educational purposes
- Research and experimentation
- Smart traffic system demonstrations

Commercial deployment requires proper authorization and compliance with local traffic regulations and privacy laws.

---

# ❤️ Credits

Built using:

- Python
- YOLOv8
- OpenCV
- EasyOCR
- DeepSort
- Streamlit
- Pandas

---

# 🤝 Contributing

Contributions are welcome!

## Steps

```bash
Fork → Clone → Create Branch → Commit → Push → Pull Request
```

---

# ⭐ Support

If you like this project, please ⭐ the repository!

---

<div align="center">

# 🚦 AI-Powered Smart Traffic Enforcement System

✨ Making Roads Smarter & Safer with Artificial Intelligence ✨

</div>
