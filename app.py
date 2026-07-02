import streamlit as st
import cv2
import requests
import numpy as np
import time

st.set_page_config(page_title="Penghitung Kendaraan Sukoharjo", layout="centered")

st.title("Pemantau & Penghitung Kendaraan")
st.subheader("CCTV Telukan Sukoharjo (Live Stream)")

# Kontainer Tampilan di Web
frame_placeholder = st.empty()
status_placeholder = st.empty()

cctv_url = "https://zmcctv.sukoharjokab.go.id/zm/cgi-bin/nph-zms?mode=jpeg&monitor=15&scale=100&maxfps=25&buffer=1000&user=user&pass=user"

# Variabel memori untuk menyimpan frame sebelumnya
if "prev_gray" not in st.session_state:
    st.session_state.prev_gray = None

jalankan = st.checkbox("Mulai Pemantauan AI", value=True)

while jalankan:
    try:
        # Menambahkan verify=False untuk melewati proteksi SSL jika sertifikat server CCTV Sukoharjo bermasalah di Linux
        response = requests.get(cctv_url, timeout=10, verify=False)
        
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Standarisasi ukuran agar ringan di web
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Proses prapemrosesan gambar
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                jumlah_kendaraan_bergerak = 0
                
                # Analisis pergerakan piksel jika frame sebelumnya sudah terekam
                if st.session_state.prev_gray is not None:
                    frame_delta = cv2.absdiff(st.session_state.prev_gray, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    
                    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in cnts:
                        if cv2.contourArea(c) < 500:
                            continue
                        (x, y, w, h) = cv2.boundingRect(c)
                        # Menggambar kotak pelacak hijau
                        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 136), 2)
                        jumlah_kendaraan_bergerak += 1
                
                # Simpan keadaan frame saat ini ke dalam memori session
                st.session_state.prev_gray = gray
                
                # Konversi BGR ke RGB agar warna di browser akurat (tidak biru/pucat)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # PAKSA TAMPILKAN: Mengirimkan gambar langsung ke placeholder tanpa peduli frame awal/akhir
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Perbarui teks status di bawah gambar
                status_placeholder.success(f"Koneksi Berhasil. Kendaraan Bergerak Terdeteksi: {jumlah_kendaraan_bergerak}")
        else:
            status_placeholder.error(f"Gagal mengambil data CCTV. Kode Status Server: {response.status_code}")
            
    except Exception as e:
        # Jika terjadi error jaringan, tampilkan pesan lognya agar mudah dilacak
        status_placeholder.warning(f"Menghubungkan ulang ke aliran CCTV... (Info: {str(e)})")
        
    # Jeda 1.2 detik sebelum melompat ke frame berikutnya agar stabil
    time.sleep(1.2)
