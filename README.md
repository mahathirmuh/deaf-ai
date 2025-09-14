# 🤖 MediaPipe Hand Landmarker

> **Deteksi landmark tangan real-time profesional menggunakan AI MediaPipe Google**  
> **Professional real-time hand landmark detection using Google's MediaPipe AI**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Language / Bahasa
- [🇮🇩 Bahasa Indonesia](#bahasa-indonesia)
- [🇺🇸 English](#english)

---

# 🇮🇩 Bahasa Indonesia

## 📋 Saran Nama Repository

### **Nama Repository yang Direkomendasikan:**
- `mediapipe-hand-landmarker`
- `realtime-hand-detection`
- `ai-hand-tracker`
- `mediapipe-hand-ai`
- `smart-hand-detector`

### **Deskripsi Repository:**
```
Sistem deteksi landmark tangan real-time profesional menggunakan AI MediaPipe Google.
Fitur pelacakan tangan 3D 21-titik, deteksi multi-tangan, dan integrasi kamera langsung
dengan OpenCV. Sempurna untuk pengenalan gerakan, pemrosesan bahasa isyarat, dan aplikasi AR.
```

## 🚀 Fitur

### **Kemampuan Inti**
- ✅ **21-Titik Landmark Tangan** - Deteksi koordinat 3D yang presisi
- ✅ **Pemrosesan Real-Time** - Feed kamera langsung dengan performa optimal
- ✅ **Deteksi Multi-Tangan** - Pelacakan simultan hingga 2 tangan
- ✅ **Akurasi Profesional** - Model MediaPipe terlatih dari Google
- ✅ **Cross-Platform** - Dukungan Windows, macOS, Linux

### **Fitur Lanjutan**
- 🎯 **Klasifikasi Tangan** - Identifikasi tangan kiri/kanan
- 📊 **Monitoring Performa** - Penghitung FPS real-time
- 🎨 **Kustomisasi Visual** - Toggle landmark dan koneksi
- 📸 **Tangkap Screenshot** - Simpan hasil deteksi
- ⏸️ **Kontrol Interaktif** - Fungsi pause/resume

## 🛠️ Instalasi

### **Prasyarat**
- Python 3.8 atau lebih tinggi
- Webcam atau perangkat kamera
- Windows 10/11, macOS 10.14+, atau Linux

### **Setup Cepat**

```bash
# Clone repository
git clone https://github.com/yourusername/mediapipe-hand-landmarker.git
cd mediapipe-hand-landmarker

# Install dependencies
pip install mediapipe opencv-python numpy

# Jalankan aplikasi
python mediapipe_hand_detector.py
```

### **Instalasi Alternatif**

```bash
# Menggunakan requirements.txt (direkomendasikan)
pip install -r requirements.txt

# Atau install satu per satu
pip install mediapipe>=0.10.0
pip install opencv-python>=4.8.0
pip install numpy>=1.21.0
```

## 🎮 Penggunaan

### **Penggunaan Dasar**

```python
from mediapipe_hand_detector import MediaPipeHandDetector

# Inisialisasi detector
detector = MediaPipeHandDetector()

# Mulai deteksi
detector.run()
```

### **Konfigurasi Lanjutan**

```python
# Konfigurasi kustom
detector = MediaPipeHandDetector(
    model_path="custom_model.task",
    max_hands=2,
    detection_confidence=0.7,
    tracking_confidence=0.5
)

# Jalankan dengan pengaturan kustom
detector.run(camera_id=1, window_name="Custom Hand Tracker")
```

### **Kontrol Interaktif**

| Tombol | Aksi |
|--------|------|
| `SPACE` | Pause/Resume deteksi |
| `S` | Simpan screenshot dengan landmark |
| `C` | Toggle koneksi landmark |
| `L` | Toggle titik landmark |
| `Q` / `ESC` | Keluar aplikasi |

## 📊 Spesifikasi Teknis

### **Akurasi Deteksi**
- **Presisi Landmark**: Akurasi sub-pixel
- **Jangkauan Deteksi**: 0.5m - 3.0m dari kamera
- **Kecepatan Pemrosesan**: 30+ FPS pada hardware modern
- **Dukungan Ukuran Tangan**: Variasi skala 20x

### **Kebutuhan Sistem**
- **CPU**: Intel i5 / AMD Ryzen 5 atau lebih baik
- **RAM**: 4GB minimum, 8GB direkomendasikan
- **Kamera**: 720p minimum, 1080p direkomendasikan
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

## 🚨 Troubleshooting

### **Masalah Umum**

**Kamera Tidak Ditemukan**
```bash
# Cek kamera yang tersedia
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
```

**Gagal Download Model**
```bash
# Download model manual
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**Masalah Performa**
- Kurangi `max_num_hands` menjadi 1
- Turunkan resolusi kamera
- Tutup aplikasi lain
- Update driver grafis

## 🤝 Kontribusi

### **Setup Development**

```bash
# Fork dan clone
git clone https://github.com/yourusername/mediapipe-hand-landmarker.git
cd mediapipe-hand-landmarker

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### **Standar Kode**
- Ikuti panduan style PEP 8
- Tambahkan docstring ke semua fungsi
- Sertakan type hints jika memungkinkan
- Tulis unit test untuk fitur baru

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Penghargaan

- **Tim Google MediaPipe** - Untuk model AI yang luar biasa
- **Komunitas OpenCV** - Untuk tools computer vision
- **Komunitas Python** - Untuk ekosistem yang menakjubkan

## 📞 Dukungan

- 📧 **Email**: support@yourproject.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/mediapipe-hand-landmarker/issues)
- 💬 **Diskusi**: [GitHub Discussions](https://github.com/yourusername/mediapipe-hand-landmarker/discussions)
- 📖 **Dokumentasi**: [Wiki](https://github.com/yourusername/mediapipe-hand-landmarker/wiki)

---

# 🇺🇸 English

## 📋 Repository Suggestions

### **Recommended Repository Names:**
- `mediapipe-hand-landmarker`
- `realtime-hand-detection`
- `ai-hand-tracker`
- `mediapipe-hand-ai`
- `smart-hand-detector`

### **Repository Description:**
```
Professional real-time hand landmark detection system using Google's MediaPipe AI. 
Features 21-point 3D hand tracking, multi-hand detection, and live camera integration 
with OpenCV. Perfect for gesture recognition, sign language processing, and AR applications.
```

## 🚀 Features

### **Core Capabilities**
- ✅ **21-Point Hand Landmarks** - Precise 3D coordinate detection
- ✅ **Real-Time Processing** - Live camera feed with optimized performance
- ✅ **Multi-Hand Detection** - Simultaneous tracking of up to 2 hands
- ✅ **Professional Accuracy** - Google's trained MediaPipe models
- ✅ **Cross-Platform** - Windows, macOS, Linux support

### **Advanced Features**
- 🎯 **Hand Classification** - Left/Right hand identification
- 📊 **Performance Monitoring** - Real-time FPS counter
- 🎨 **Visual Customization** - Toggle landmarks and connections
- 📸 **Screenshot Capture** - Save detection results
- ⏸️ **Interactive Controls** - Pause/resume functionality

## 🛠️ Installation

### **Prerequisites**
- Python 3.8 or higher
- Webcam or camera device
- Windows 10/11, macOS 10.14+, or Linux

### **Quick Setup**

```bash
# Clone the repository
git clone https://github.com/yourusername/mediapipe-hand-landmarker.git
cd mediapipe-hand-landmarker

# Install dependencies
pip install mediapipe opencv-python numpy

# Run the application
python mediapipe_hand_detector.py
```

### **Alternative Installation**

```bash
# Using requirements.txt (recommended)
pip install -r requirements.txt

# Or install individually
pip install mediapipe>=0.10.0
pip install opencv-python>=4.8.0
pip install numpy>=1.21.0
```

## 🎮 Usage

### **Basic Usage**

```python
from mediapipe_hand_detector import MediaPipeHandDetector

# Initialize detector
detector = MediaPipeHandDetector()

# Start detection
detector.run()
```

### **Advanced Configuration**

```python
# Custom configuration
detector = MediaPipeHandDetector(
    model_path="custom_model.task",
    max_hands=2,
    detection_confidence=0.7,
    tracking_confidence=0.5
)

# Run with custom settings
detector.run(camera_id=1, window_name="Custom Hand Tracker")
```

### **Interactive Controls**

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume detection |
| `S` | Save screenshot with landmarks |
| `C` | Toggle landmark connections |
| `L` | Toggle landmark points |
| `Q` / `ESC` | Exit application |

## 📊 Technical Specifications

### **Detection Accuracy**
- **Landmark Precision**: Sub-pixel accuracy
- **Detection Range**: 0.5m - 3.0m from camera
- **Processing Speed**: 30+ FPS on modern hardware
- **Hand Size Support**: 20x scale variation

### **System Requirements**
- **CPU**: Intel i5 / AMD Ryzen 5 or better
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: 720p minimum, 1080p recommended
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

## 🏗️ Architecture

### **Core Components**

```
📦 MediaPipe Hand Landmarker
├── 🧠 MediaPipeHandDetector     # Main detection class
├── 📹 Camera Integration        # OpenCV camera handling
├── 🎨 Visualization Engine      # Landmark rendering
├── ⚙️ Configuration Manager     # Settings and options
└── 📊 Performance Monitor       # FPS and metrics
```

### **Data Flow**

```
Camera Feed → MediaPipe AI → Landmark Detection → Visualization → Display
     ↓              ↓              ↓               ↓
  Frame Capture → Hand Analysis → 21-Point Coords → Real-time Render
```

## 🔧 Configuration

### **Detection Parameters**

```python
# Confidence thresholds
min_detection_confidence = 0.5    # Hand detection sensitivity
min_tracking_confidence = 0.5     # Tracking stability
min_presence_confidence = 0.5     # Hand presence detection

# Performance settings
max_num_hands = 2                 # Maximum hands to detect
model_complexity = 1              # Model accuracy vs speed
```

### **Visual Settings**

```python
# Landmark styling
landmark_color = (0, 255, 0)      # Green landmarks
connection_color = (255, 0, 0)    # Red connections
landmark_size = 3                 # Point radius
connection_thickness = 2          # Line thickness
```

## 🚨 Troubleshooting

### **Common Issues**

**Camera Not Found**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
```

**Model Download Failed**
```bash
# Manual model download
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**Performance Issues**
- Reduce `max_num_hands` to 1
- Lower camera resolution
- Close other applications
- Update graphics drivers

## 🤝 Contributing

### **Development Setup**

```bash
# Fork and clone
git clone https://github.com/yourusername/mediapipe-hand-landmarker.git
cd mediapipe-hand-landmarker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### **Code Standards**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google MediaPipe Team** - For the incredible AI models
- **OpenCV Community** - For computer vision tools
- **Python Community** - For the amazing ecosystem

## 📞 Support

- 📧 **Email**: support@yourproject.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/mediapipe-hand-landmarker/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/mediapipe-hand-landmarker/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/yourusername/mediapipe-hand-landmarker/wiki)

---

<div align="center">
  <strong>🇮🇩 Dibuat dengan ❤️ menggunakan AI MediaPipe Google</strong><br>
  <strong>🇺🇸 Built with ❤️ using Google MediaPipe AI</strong>
</div>