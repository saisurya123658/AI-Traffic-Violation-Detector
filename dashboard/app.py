
import streamlit as st
from pathlib import Path
import os
import cv2
import time
import json
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Traffic Violation Dashboard", layout="wide")
st.title("🚦 AI Traffic Violation Dashboard (Demo)")
st.caption("Upload a video, and the AI system will detect traffic violations automatically.")

# -------------------------------
# Upload Video
# -------------------------------
uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
process_btn = st.button("🚀 Process Video")

# -------------------------------
# Initialize Models
# -------------------------------
@st.cache_resource
def load_models():
    # Make sure the models exist at these paths
    vehicle_model_path = r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\models\yolov8n.pt"
    helmet_model_path  = r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\models\helmet.pt"

    if not os.path.exists(vehicle_model_path):
        st.error(f"Vehicle model not found at {vehicle_model_path}")
        return None, None, None, None
    if not os.path.exists(helmet_model_path):
        st.warning(f"Helmet model not found at {helmet_model_path} (helmet violations will be skipped)")

    model = YOLO(vehicle_model_path)
    helmet_model = YOLO(helmet_model_path) if os.path.exists(helmet_model_path) else None
    tracker = DeepSort(max_age=30)
    reader = easyocr.Reader(['en'])
    return model, helmet_model, tracker, reader

model, helmet_model, tracker, reader = load_models()

# -------------------------------
# Violations Storage
# -------------------------------
violations_dir = Path("violations")
violations_dir.mkdir(exist_ok=True)
violations_json = violations_dir / "violations.json"
if not violations_json.exists():
    with open(violations_json, "w") as f:
        json.dump([], f, indent=4)

# -------------------------------
# Video Processing
# -------------------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_violations = []
    prev_centroids = {}
    line_y = 300
    ocr_skip = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]
        frame_time = time.time()

        # Detect vehicles
        results = model(frame, imgsz=640)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            label = int(cls)
            if label in [0,1,2,3,5,7]:  # vehicle/person classes
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)
        signal_state = "RED"  # dummy demo

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cX, cY = (int((x1+x2)/2), int((y1+y2)/2))
            vehicle_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].copy()
            if vehicle_crop.size == 0:
                prev_centroids[tid] = (cX, cY)
                continue

            # OCR every few frames
            lp = "UNKNOWN"
            if frame_count % ocr_skip == 0:
                lp_texts = reader.readtext(vehicle_crop)
                if lp_texts:
                    lp = lp_texts[0][1]

            # Signal violation
            violation_type = None
            if tid in prev_centroids:
                _, prevY = prev_centroids[tid]
                if prevY < line_y <= cY and signal_state=="RED":
                    violation_type = "signal_violation"

            # Save violation
            if violation_type:
                fname = violations_dir / f"{violation_type}_{int(frame_time)}_{tid}.jpg"
                cv2.imwrite(str(fname), vehicle_crop)
                all_violations.append({
                    "violation_type": violation_type,
                    "track_id": tid,
                    "license_plate": lp,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": str(fname),
                    "clip_path": None
                })

            prev_centroids[tid] = (cX, cY)

    cap.release()
    # Save JSON
    with open(violations_json, "w") as f:
        json.dump(all_violations, f, indent=4)

    return all_violations

# -------------------------------
# Run Processing
# -------------------------------
if process_btn and uploaded_video is not None:
    temp_file = "temp_video.mp4"
    with open(temp_file, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.info("Processing video... this may take a while.")
    violations = process_video(temp_file)
    st.success(f"✅ Processing finished! {len(violations)} violations detected.")
    os.remove(temp_file)  # cleanup

# -------------------------------
# Display Violations
# -------------------------------
if violations_json.exists():
    with open(violations_json, "r") as f:
        data = json.load(f)
    if data:
        st.subheader("📸 Detected Violations")
        for ev in reversed(data):
            cols = st.columns([1,2])
            with cols[0]:
                img_path = Path(ev["image_path"])
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
            with cols[1]:
                st.markdown(f"""
                **Violation Type:** {ev.get('violation_type','Unknown')}  
                **Track ID:** {ev.get('track_id','Unknown')}  
                **License Plate:** {ev.get('license_plate','Unknown')}  
                **Timestamp:** {ev.get('timestamp','Unknown')}  
                """)
            st.divider()


