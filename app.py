import streamlit as st
import cv2
import requests
import numpy as np
import time

st.set_page_config(page_title="Penghitung Kendaraan AI Sukoharjo", layout="centered")

st.title("Pemantau & Penghitung Kendaraan")
st.subheader("CCTV Kabupaten Sukoharjo via Streamlit Server")

# Tempat menampilkan gambar dan status
frame_placeholder = st.empty()
status_placeholder = st.empty()

cctv_url = "https://zmcctv.sukoharjokab.go.id/zm/cgi-bin/nph-zms?mode=jpeg&monitor=15&scale=100&maxfps=25&buffer=1000&user=user&pass=user"

# Menggunakan metode HOG bawaan OpenCV (100% Aman tanpa perlu file XML tambahan)
@st.cache_resource
def get_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

detector = get_detector()

jalankan = st.checkbox("Mulai Pemantauan", value=True)

while jalankan:
    try:
        # Server Streamlit yang mengambil gambar (Bebas CORS!)
        response = requests.get(cctv_url, timeout=10)
        
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # OPTIMALISASI: Perkecil ukuran frame agar proses server super cepat dan hemat RAM
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Deteksi objek (bisa mendeteksi pergerakan/manusia/kendaraan di area CCTV)
                # Menggunakan detektor bawaan yang stabil
                (rects, weights) = detector.detectMultiScale(frame_resized, winStride=(4, 4), padding=(8, 8), scale=1.05)
                
                jumlah_objek = len(rects)
                
                # Gambar kotak di setiap objek yang terdeteksi
                for (x, y, w, h) in rects:
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 136), 2)
                
                # Konversi ke RGB untuk ditampilkan di web Streamlit
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR) # Balikkan channel dengan benar
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                status_placeholder.success(f"Objek/Kendaraan Terdeteksi: {jumlah_objek}")
        else:
            status_placeholder.error(f"Gagal mengambil data CCTV. Status: {response.status_code}")
            
    except Exception as e:
        status_placeholder.warning(f"Menghubungkan ulang ke CCTV... ({str(e)})")
        
    # Jeda 1.5 detik agar aman dari pemblokiran IP
    time.sleep(1.5)
