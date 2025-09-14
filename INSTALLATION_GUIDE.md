# 🚀 Installation & Setup Guide

## Quick Start (TL;DR)

```bash
# Clone the repository
git clone <repository-url>
cd mediapipe-hand-landmarker

# Install dependencies
pip install -r requirements.txt

# Run the application
python mediapipe_hand_detector.py
```

---

## 📋 System Requirements

### **Minimum Requirements**
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Intel i5 / AMD Ryzen 5 or equivalent
- **Camera**: USB webcam or built-in camera
- **Storage**: 500MB free space

### **Recommended Requirements**
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 - 3.11 (optimal compatibility)
- **RAM**: 16GB for optimal performance
- **CPU**: Intel i7 / AMD Ryzen 7 or better
- **GPU**: Optional, for enhanced performance
- **Camera**: HD webcam (1080p) for best results
- **Storage**: 2GB free space (including models)

---

## 🔧 Installation Methods

### **Method 1: Standard Installation (Recommended)**

#### **Step 1: Python Environment Setup**

**Windows:**
```powershell
# Check Python version
python --version

# Create virtual environment
python -m venv mediapipe_env

# Activate virtual environment
mediapipe_env\Scripts\activate
```

**macOS/Linux:**
```bash
# Check Python version
python3 --version

# Create virtual environment
python3 -m venv mediapipe_env

# Activate virtual environment
source mediapipe_env/bin/activate
```

#### **Step 2: Clone Repository**
```bash
# Clone the repository
git clone <repository-url>
cd mediapipe-hand-landmarker

# Verify files
ls -la  # macOS/Linux
dir     # Windows
```

#### **Step 3: Install Dependencies**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep mediapipe
```

#### **Step 4: Test Installation**
```bash
# Run the application
python mediapipe_hand_detector.py

# Expected output:
# MediaPipe Hand Landmarker initialized successfully!
# Camera started successfully
# Press 'q' to quit, 'h' to toggle help
```

---

### **Method 2: Development Installation**

#### **For Contributors and Developers**

```bash
# Clone with development branch
git clone -b develop <repository-url>
cd mediapipe-hand-landmarker

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/

# Run linting
flake8 .
black .
```

---

### **Method 3: Docker Installation**

#### **Using Docker (Advanced Users)**

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose camera device
CMD ["python", "mediapipe_hand_detector.py"]
```

```bash
# Build and run Docker container
docker build -t mediapipe-hand-landmarker .
docker run --device=/dev/video0 -it mediapipe-hand-landmarker
```

---

## 🔍 Dependency Details

### **Core Dependencies**

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `mediapipe` | ≥0.10.0 | AI hand detection engine | ~50MB |
| `opencv-python` | ≥4.5.0 | Computer vision and camera handling | ~30MB |
| `numpy` | ≥1.21.0 | Numerical computations | ~15MB |

### **Optional Dependencies**

| Package | Version | Purpose | Installation |
|---------|---------|---------|-------------|
| `psutil` | ≥5.8.0 | Performance monitoring | `pip install psutil` |
| `matplotlib` | ≥3.5.0 | Data visualization | `pip install matplotlib` |
| `pillow` | ≥8.0.0 | Image processing | `pip install pillow` |

### **Development Dependencies**

| Package | Version | Purpose |
|---------|---------|----------|
| `pytest` | ≥6.0.0 | Unit testing |
| `black` | ≥22.0.0 | Code formatting |
| `flake8` | ≥4.0.0 | Code linting |
| `mypy` | ≥0.950 | Type checking |

---

## 🎯 Configuration Setup

### **Camera Configuration**

#### **1. Test Camera Access**
```python
# test_camera.py
import cv2

def test_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Camera {camera_id} not accessible")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"✅ Camera {camera_id} working - Resolution: {frame.shape}")
        cap.release()
        return True
    else:
        print(f"❌ Camera {camera_id} not reading frames")
        cap.release()
        return False

# Test multiple cameras
for i in range(3):
    test_camera(i)
```

#### **2. Camera Permissions**

**Windows:**
- Go to Settings → Privacy & Security → Camera
- Enable "Allow apps to access your camera"
- Enable "Allow desktop apps to access your camera"

**macOS:**
- Go to System Preferences → Security & Privacy → Camera
- Check the box next to Terminal or your IDE

**Linux:**
```bash
# Check camera devices
ls /dev/video*

# Check permissions
ls -l /dev/video0

# Add user to video group (if needed)
sudo usermod -a -G video $USER
```

### **Performance Tuning**

#### **1. Optimize for Your Hardware**

```python
# config.py - Create this file for custom settings
CONFIG = {
    # For slower hardware
    'LOW_PERFORMANCE': {
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5,
        'model_complexity': 0,
        'max_num_hands': 1,
    },
    
    # For standard hardware
    'BALANCED': {
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'model_complexity': 1,
        'max_num_hands': 2,
    },
    
    # For high-end hardware
    'HIGH_PERFORMANCE': {
        'min_detection_confidence': 0.3,
        'min_tracking_confidence': 0.3,
        'model_complexity': 1,
        'max_num_hands': 2,
    }
}
```

#### **2. Memory Optimization**

```python
# Add to mediapipe_hand_detector.py
import gc
import os

# Optimize memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs
gc.set_threshold(700, 10, 10)  # Optimize garbage collection
```

---

## 🚨 Troubleshooting Guide

### **Common Issues & Solutions**

#### **❌ Issue 1: "No module named 'mediapipe'"**

**Symptoms:**
```
ModuleNotFoundError: No module named 'mediapipe'
```

**Solutions:**
```bash
# Solution 1: Check virtual environment
which python  # Should show virtual env path
pip list | grep mediapipe

# Solution 2: Reinstall mediapipe
pip uninstall mediapipe
pip install mediapipe

# Solution 3: Check Python version
python --version  # Should be 3.8+
```

---

#### **❌ Issue 2: "Camera not found" or "Cannot open camera"**

**Symptoms:**
```
[ERROR] Failed to open camera 0
Camera initialization failed
```

**Solutions:**

**Step 1: Check camera availability**
```python
# Run this test script
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
    else:
        print(f"Camera {i}: Not available")
```

**Step 2: Try different camera IDs**
```bash
# Run with different camera ID
python mediapipe_hand_detector.py --camera 1
python mediapipe_hand_detector.py --camera 2
```

**Step 3: Check camera permissions (see Configuration Setup above)**

**Step 4: Restart camera service**
```bash
# Windows
# Restart in Device Manager

# Linux
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo

# macOS
# Restart the application
```

---

#### **❌ Issue 3: "Low FPS" or "Laggy performance"**

**Symptoms:**
- FPS counter shows < 15 FPS
- Jerky or delayed hand tracking
- High CPU usage

**Solutions:**

**Step 1: Reduce camera resolution**
```python
# Add to mediapipe_hand_detector.py
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**Step 2: Optimize MediaPipe settings**
```python
# Use lower complexity model
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    num_hands=1,  # Reduce from 2 to 1
    min_hand_detection_confidence=0.7,  # Increase threshold
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
    result_callback=self._result_callback
)
```

**Step 3: Close other applications**
- Close video conferencing apps (Zoom, Teams, etc.)
- Close other camera applications
- Close resource-intensive programs

---

#### **❌ Issue 4: "Model download failed"**

**Symptoms:**
```
[ERROR] Failed to download hand_landmarker.task
HTTPError: 404 Client Error
```

**Solutions:**

**Step 1: Manual model download**
```bash
# Create models directory
mkdir models
cd models

# Download model manually
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Or use curl
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**Step 2: Check internet connection**
```bash
# Test connectivity
ping google.com
curl -I https://storage.googleapis.com
```

**Step 3: Use local model path**
```python
# Run with local model
detector = MediaPipeHandDetector(model_path="./models/hand_landmarker.task")
```

---

#### **❌ Issue 5: "ImportError: DLL load failed" (Windows)**

**Symptoms:**
```
ImportError: DLL load failed while importing cv2
```

**Solutions:**

**Step 1: Install Visual C++ Redistributables**
- Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Install and restart

**Step 2: Reinstall OpenCV**
```bash
pip uninstall opencv-python
pip install opencv-python
```

**Step 3: Try alternative OpenCV build**
```bash
pip install opencv-python-headless
```

---

#### **❌ Issue 6: "Hand detection not working"**

**Symptoms:**
- Camera works but no hand landmarks appear
- No error messages
- FPS counter working

**Solutions:**

**Step 1: Check lighting conditions**
- Ensure good lighting on hands
- Avoid backlighting
- Use consistent background

**Step 2: Adjust detection thresholds**
```python
# Lower confidence thresholds
min_hand_detection_confidence=0.3,  # Default: 0.5
min_hand_presence_confidence=0.3,   # Default: 0.5
min_tracking_confidence=0.3,        # Default: 0.5
```

**Step 3: Check hand positioning**
- Keep hands within camera frame
- Ensure hands are clearly visible
- Try different hand poses

---

### **Performance Optimization**

#### **System Optimization Checklist**

- [ ] **Close unnecessary applications**
- [ ] **Use wired camera connection** (avoid USB hubs)
- [ ] **Ensure adequate lighting**
- [ ] **Update graphics drivers**
- [ ] **Use SSD storage** (if available)
- [ ] **Enable hardware acceleration** (if supported)

#### **Code Optimization Tips**

```python
# 1. Reduce frame processing
skip_frames = 2  # Process every 3rd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % skip_frames == 0:
        # Process frame
        pass

# 2. Resize frames for processing
processing_width = 640
processing_height = 480
frame_resized = cv2.resize(frame, (processing_width, processing_height))

# 3. Use threading for display
import threading
from queue import Queue

display_queue = Queue(maxsize=2)

def display_thread():
    while True:
        if not display_queue.empty():
            frame = display_queue.get()
            cv2.imshow('MediaPipe Hand Landmarker', frame)
```

---

## 🔧 Advanced Configuration

### **Environment Variables**

```bash
# Set environment variables for optimization
export MEDIAPIPE_DISABLE_GPU=1          # Disable GPU (if causing issues)
export TF_CPP_MIN_LOG_LEVEL=2           # Reduce TensorFlow logging
export OPENCV_VIDEOIO_PRIORITY_MSMF=0   # Windows: Use DirectShow instead of Media Foundation
```

### **Custom Model Configuration**

```python
# Use custom trained model
custom_model_path = "./models/custom_hand_model.task"
detector = MediaPipeHandDetector(model_path=custom_model_path)

# Configure for specific use case
if use_case == "sign_language":
    config = {
        'min_detection_confidence': 0.8,
        'min_tracking_confidence': 0.8,
        'max_num_hands': 2,
    }
elif use_case == "gesture_control":
    config = {
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.7,
        'max_num_hands': 1,
    }
```

---

## 📞 Getting Help

### **Support Channels**

1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check API_DOCUMENTATION.md
3. **Community**: Join discussions in GitHub Discussions
4. **Stack Overflow**: Tag questions with `mediapipe` and `hand-detection`

### **Before Asking for Help**

**Include this information:**

```bash
# System information
python --version
pip list | grep -E "mediapipe|opencv|numpy"

# OS information
# Windows
systeminfo | findstr /B /C:"OS Name" /C:"OS Version"

# macOS
sw_vers

# Linux
lsb_release -a

# Camera information
# Run the camera test script above
```

**Error logs:**
- Full error traceback
- Steps to reproduce
- Expected vs actual behavior

---

## ✅ Verification Checklist

### **Installation Verification**

- [ ] Python 3.8+ installed and accessible
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] Camera permissions granted
- [ ] Camera accessible and working
- [ ] MediaPipe model downloaded
- [ ] Application runs without errors
- [ ] Hand detection working properly
- [ ] FPS > 20 on target hardware
- [ ] All controls (q, h, etc.) working

### **Performance Verification**

- [ ] Smooth hand tracking (no jitter)
- [ ] Acceptable latency (< 100ms)
- [ ] Stable FPS under normal conditions
- [ ] Memory usage reasonable (< 1GB)
- [ ] CPU usage acceptable (< 50% on modern hardware)

---

## 🎉 Success!

If you've completed all the steps above, you should have a fully functional MediaPipe Hand Landmarker system running on your machine. The application should display:

- ✅ Real-time camera feed
- ✅ Hand landmark detection and tracking
- ✅ FPS counter and performance metrics
- ✅ Interactive controls and help system

Enjoy exploring the capabilities of advanced hand tracking technology! 🚀

---

## 📚 Next Steps

1. **Explore the API**: Check out `API_DOCUMENTATION.md`
2. **Understand the Architecture**: Read `ARCHITECTURE.md`
3. **Customize the Application**: Modify settings for your use case
4. **Build Something Amazing**: Use the hand tracking in your projects!

---

*For additional support and updates, please visit the project repository and documentation.*