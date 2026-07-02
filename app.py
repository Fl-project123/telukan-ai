import streamlit as st
import cv2
import requests
import numpy as np
import time

st.set_page_config(page_title="Penghitung Kendaraan Sukoharjo", layout="centered")

st.title("Pemantau & Penghitung Kendaraan")
st.subheader("CCTV Telukan Sukoharjo via Perubahan Piksel (Bebas Error)")

# Kontainer Tampilan
frame_placeholder = st.empty()
status_placeholder = st.empty()

cctv_url = "https://zmcctv.sukoharjokab.go.id/zm/cgi-bin/nph-zms?mode=jpeg&monitor=15&scale=100&maxfps=25&buffer=1000&user=user&pass=user"

# Variabel untuk menyimpan frame sebelumnya (Background)
if "prev_gray" not in st.session_state:
    st.session_state.prev_gray = None

jalankan = st.checkbox("Mulai Pemantauan", value=True)

while jalankan:
    try:
        # Ambil frame dari CCTV (Lolos CORS karena ditembak dari server Streamlit)
        response = requests.get(cctv_url, timeout=10)
        
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 1. Perkecil resolusi agar performa server sangat cepat
                frame_resized = cv2.resize(frame, (640, 480))
                
                # 2. Ubah ke warna Grayscale dan beri sedikit Blur untuk mengurangi noise
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                jumlah_kendaraan_bergerak = 0
                
                # 3. Jika ini bukan frame pertama, bandingkan dengan frame sebelumnya
                if st.session_state.prev_gray is not None:
                    # Hitung perbedaan absolut antar pixel
                    frame_delta = cv2.absdiff(st.session_state.prev_gray, gray)
                    # Berikan ambang batas (threshold) agar piksel yang berubah menjadi putih (objek)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    
                    # Cari kontur/bentuk dari piksel yang bergerak tersebut
                    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in cnts:
                        # Abaikan kontur yang terlalu kecil (seperti daun bergerak atau noise kamera)
                        if cv2.contourArea(c) < 500:
                            continue
                            
                        # Ambil koordinat kotak untuk kontur yang besar (kendaraan bergerak)
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 136), 2)
                        jumlah_kendaraan_bergerak += 1
                
                # Simpan frame saat ini untuk dibandingkan dengan frame berikutnya
                st.session_state.prev_gray = gray
                
                # Tampilkan ke Dashboard Streamlit
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                if st.session_state.prev_gray is None:
                    status_placeholder.info("Menginisialisasi sistem pemantauan...")
                else:
                    status_placeholder.success(f"Kendaraan Bergerak Terdeteksi: {jumlah_kendaraan_bergerak}")
        else:
            status_placeholder.error(f"Gagal mengambil data CCTV. Status: {response.status_code}")
            
    except Exception as e:
        status_placeholder.warning("Menghubungkan ulang ke aliran CCTV...")
        
    time.sleep(1.0)
