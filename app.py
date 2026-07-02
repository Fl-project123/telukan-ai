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

# Memuat model Haar Cascade bawaan OpenCV untuk mendeteksi mobil (Sangat ringan & tanpa DNN)
@st.cache_resource
def load_cascade():
    # Mengunduh file pendeteksi mobil resmi milik OpenCV
    cascade_url = "https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml"
    response = requests.get(cascade_url)
    with open("cars.xml", "wb") as f:
        f.write(response.content)
    return cv2.CascadeClassifier("cars.xml")

car_cascade = load_cascade()

jalankan = st.checkbox("Mulai Pemantauan", value=True)

while jalankan:
    try:
        # Server Streamlit yang mengambil gambar (Bukan browser Anda, jadi BEBAS CORS!)
        response = requests.get(cctv_url, timeout=10)
        
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Mengubah ke keabuan (Gray) agar pemrosesan super cepat
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Deteksi mobil menggunakan Cascade
                cars = car_cascade.detectMultiScale(gray, 1.1, 2)
                jumlah_kendaraan = len(cars)
                
                # Gambar kotak di setiap mobil yang terdeteksi
                for (x, y, w, h) in cars:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 136), 2)
                
                # Konversi ke RGB untuk ditampilkan di web Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                status_placeholder.success(f"Kendaraan Terdeteksi: {jumlah_kendaraan}")
        else:
            status_placeholder.error(f"Gagal mengambil data CCTV. Status: {response.status_code}")
            
    except Exception as e:
        status_placeholder.warning("Menghubungkan ulang ke CCTV...")
        
    # Jeda 1.5 detik agar server CCTV tidak memblokir IP server Streamlit
    time.sleep(1.5)
