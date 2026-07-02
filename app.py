import streamlit as st
import cv2
import numpy as np
import time

st.set_page_config(page_title="Penghitung Kendaraan Sukoharjo", layout="centered")

st.title("Pemantau & Penghitung Kendaraan")
st.subheader("Simulasi Sistem Penghitung Kendaraan AI (FL Project)")

frame_placeholder = st.empty()
status_placeholder = st.empty()

# Menggunakan video lalu lintas publik sebagai pengganti stream yang terblokir firewall
video_url = "https://assets.mixkit.co/videos/preview/mixkit-traffic-on-a-highway-at-night-40176-large.mp4"

if "prev_gray" not in st.session_state:
    st.session_state.prev_gray = None

jalankan = st.checkbox("Mulai Pemantauan AI", value=True)

# Membuka file video stream
cap = cv2.VideoCapture(video_url)

while jalankan and cap.isOpened():
    ret, frame = cap.read()
    
    # Jika video habis, putar kembali dari awal (loop)
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
        
    try:
        frame_resized = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        jumlah_kendaraan_bergerak = 0
        
        if st.session_state.prev_gray is not None:
            frame_delta = cv2.absdiff(st.session_state.prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                if cv2.contourArea(c) < 400:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 136), 2)
                jumlah_kendaraan_bergerak += 1
        
        st.session_state.prev_gray = gray
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        status_placeholder.success(f"Sistem Aktif. Kendaraan Terdeteksi: {jumlah_kendaraan_bergerak}")
        
    except Exception as e:
        status_placeholder.warning(f"Error pemrosesan: {str(e)}")
        
    # Jeda kecil agar gerakan video terlihat natural di browser
    time.sleep(0.03)

cap.release()
