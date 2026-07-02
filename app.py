import streamlit as st
import cv2
import requests
import numpy as np
import time

st.set_page_config(page_title="Penghitung Kendaraan AI Sukoharjo", layout="centered")

st.title("Pemantau & Penghitung Kendaraan AI")
st.subheader("CCTV Kabupaten Sukoharjo")

# Kontainer untuk menampilkan gambar
frame_placeholder = st.empty()
status_placeholder = st.empty()

cctv_url = "https://zmcctv.sukoharjokab.go.id/zm/cgi-bin/nph-zms?mode=jpeg&monitor=15&scale=100&maxfps=25&buffer=1000&user=user&pass=user"

# Memuat detektor objek bawaan OpenCV yang sangat ringan (MobileNet-SSD)
# Server akan mengunduh file konfigurasi ini secara otomatis
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(
        cv2.samples.findFile("proto.txt", required=False) or "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        cv2.samples.findFile("model.caffemodel", required=False) or "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
    )
    return net

try:
    net = load_model()
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    # Kelas kendaraan yang ingin dihitung: bus (6), car (7), motorbike (14)
    TARGET_CLASSES = [6, 7, 14]
except Exception as e:
    st.error(f"Gagal memuat model AI: {e}")
    net = None

# Tombol untuk mengontrol jalannya sistem
jalankan = st.checkbox("Mulai Pemantauan AI", value=True)

while jalankan:
    try:
        # Request langsung dari server Streamlit (bukan dari laptop Anda, jadi lolos CORS!)
        response = requests.get(cctv_url, timeout=10)
        
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                h, w = frame.shape[:2]
                jumlah_kendaraan = 0
                
                if net is not None:
                    # Proses AI ringan
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.30: # Threshold akurasi 30%
                            idx = int(detections[0, 0, i, 1])
                            if idx in TARGET_CLASSES:
                                jumlah_kendaraan += 1
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")
                                
                                # Gambar kotak di kendaraan
                                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 136), 2)
                
                # Ubah warna BGR OpenCV ke RGB Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                status_placeholder.success(f"Kendaraan Terdeteksi: {jumlah_kendaraan}")
                
        else:
            status_placeholder.error(f"Koneksi CCTV terputus. Status: {response.status_code}")
            
    except Exception as e:
        status_placeholder.warning(f"Menunggu frame baru... ({e})")
        
    time.sleep(1.5) # Jeda agar server tidak memblokir IP
