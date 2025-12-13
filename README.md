# 🚦 AI Traffic Violation Detector & Dashboard

An AI-powered computer vision system that automatically detects traffic rule violations from CCTV or recorded video footage and displays them on an interactive dashboard for traffic authorities.

---

## 📌 Project Overview

The **AI Traffic Violation Detector & Dashboard** is designed to reduce manual traffic monitoring by leveraging **deep learning and computer vision**.  
It detects common traffic violations, extracts license plate numbers, and logs all incidents with visual evidence into a centralized dashboard.

This system helps improve **road safety**, **law enforcement efficiency**, and supports **Smart City initiatives**.

---

## ❗ Problem Statement

Traditional traffic monitoring relies heavily on:
- Manual observation by traffic police
- Basic surveillance cameras without intelligence

This leads to:
- Missed violations
- Human error
- Delayed enforcement
- Inefficient use of manpower

---

## ✅ Proposed Solution

Our system automatically:
- Detects vehicles, riders, and pedestrians
- Identifies traffic violations using AI
- Extracts license plate numbers using OCR
- Logs violations with timestamp and image evidence
- Displays everything in a real-time dashboard

---

## 🚨 Detected Traffic Violations

- 🚦 Signal Jumping
- 🪖 Helmetless Riding
- 🚗 Overspeeding
- 🔁 Wrong Lane Driving
- 👨‍👩‍👦 Triple Riding

---

## 🧠 System Architecture (High Level)

1. Video Input (CCTV / Recorded Video)
2. Frame-by-frame processing using OpenCV
3. Object Detection using YOLOv8
4. Vehicle Tracking using DeepSort
5. License Plate Recognition using EasyOCR
6. Violation Logging (JSON)
7. Dashboard Visualization using Streamlit

---

## 🛠️ Technology Stack

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

## 📊 Dashboard Features

- 🌙 Dark-themed user interface
- 🔍 Filters by violation type
- 👮 Officer login authentication
- 📸 Image evidence for each violation
- 🤖 Chatbot to query data (e.g., *"show red car violations"*)

---

## ▶️ Demo Flow

1. Upload a traffic video
2. System processes the video frame by frame
3. Vehicles and riders are detected and tracked
4. Rule violations are identified automatically
5. Evidence is captured (image + metadata)
6. Dashboard updates in real-time for officer review

---

## 📈 Impact & Use Cases

- Reduces manual workload of traffic police
- Enables evidence-based enforcement
- Improves road safety
- Suitable for:
  - Smart Cities
  - Traffic Command Centers
  - Highway Surveillance Systems

---

## 🚀 Future Enhancements

- 🔴 Live CCTV feed integration
- 💳 Automatic e-challan generation
- 🌐 Cloud backend for city-wide access
- 🇮🇳 Improved license plate OCR for Indian regional plates
- 📱 Mobile dashboard support

---

## 🧾 One-Line Summary

> **A smart AI assistant for traffic monitoring that detects violations in real-time, logs evidence automatically, and simplifies enforcement through an interactive dashboard.**

---

## 👨‍💻 Team & Contributions

This project was developed as part of a **hackathon / academic innovation project** focused on AI-driven public safety solutions.

---

## 📜 License

This project is for **educational and research purposes**.  
Commercial deployment requires proper authorization and compliance with local laws.

---

⭐ If you like this project, don’t forget to **star the repository**!
