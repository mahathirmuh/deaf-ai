# 📚 API Documentation

## MediaPipeHandDetector Class

### Overview
The `MediaPipeHandDetector` class provides a high-level interface for real-time hand landmark detection using Google's MediaPipe AI models.

### Class Definition

```python
class MediaPipeHandDetector:
    """
    Professional MediaPipe-based hand landmark detector with real-time processing capabilities.
    
    This class encapsulates Google's MediaPipe Hand Landmarker for detecting and tracking
    hand landmarks in real-time video streams with high accuracy and performance.
    """
```

---

## Constructor

### `__init__(self, model_path=None)`

Initializes the MediaPipe Hand Landmarker detector.

**Parameters:**
- `model_path` (str, optional): Path to custom MediaPipe model file (.task)
  - If None, downloads and uses the default Google model
  - Default: `None`

**Example:**
```python
# Using default model (auto-download)
detector = MediaPipeHandDetector()

# Using custom model
detector = MediaPipeHandDetector(model_path="custom_model.task")
```

**Raises:**
- `RuntimeError`: If MediaPipe initialization fails
- `FileNotFoundError`: If custom model path doesn't exist

---

## Public Methods

### `run(self, camera_id=0, window_name="MediaPipe Hand Landmarker")`

Starts the real-time hand detection with camera input.

**Parameters:**
- `camera_id` (int): Camera device index
  - Default: `0` (primary camera)
  - Range: `0-9` (system dependent)
- `window_name` (str): Display window title
  - Default: `"MediaPipe Hand Landmarker"`

**Returns:**
- `None`

**Example:**
```python
# Basic usage
detector.run()

# Custom camera and window
detector.run(camera_id=1, window_name="Hand Tracker Pro")
```

**Interactive Controls:**
- `SPACE`: Pause/Resume detection
- `S`: Save screenshot
- `C`: Toggle connections
- `L`: Toggle landmarks
- `Q/ESC`: Exit

**Raises:**
- `RuntimeError`: If camera initialization fails
- `cv2.error`: If OpenCV operations fail

---

### `cleanup(self)`

Cleans up resources and closes the detector.

**Parameters:**
- None

**Returns:**
- `None`

**Example:**
```python
detector = MediaPipeHandDetector()
try:
    detector.run()
finally:
    detector.cleanup()  # Ensure proper cleanup
```

---

## Private Methods

### `_setup_mediapipe(self, model_path=None)`

Configures and initializes the MediaPipe Hand Landmarker.

**Parameters:**
- `model_path` (str, optional): Path to model file

**Returns:**
- `None`

**Features:**
- Automatic model downloading if not present
- Configures live stream mode for real-time processing
- Sets up result callback for asynchronous processing

---

### `_result_callback(self, result, output_image, timestamp_ms)`

Processes MediaPipe detection results asynchronously.

**Parameters:**
- `result` (HandLandmarkerResult): Detection results from MediaPipe
- `output_image` (mp.Image): Processed image frame
- `timestamp_ms` (int): Frame timestamp in milliseconds

**Returns:**
- `None`

**Processing:**
- Extracts hand landmarks and handedness
- Stores results for visualization
- Thread-safe result handling

---

### `_draw_landmarks(self, image, landmarks, handedness)`

Renders hand landmarks and connections on the image.

**Parameters:**
- `image` (numpy.ndarray): Input image frame
- `landmarks` (list): Hand landmark coordinates
- `handedness` (list): Hand classification (Left/Right)

**Returns:**
- `numpy.ndarray`: Annotated image with landmarks

**Visual Features:**
- 21-point landmark visualization
- Hand connection lines
- Left/Right hand labels
- Customizable colors and styles

---

### `_calculate_fps(self)`

Calculates and updates the frames per second counter.

**Parameters:**
- None

**Returns:**
- `float`: Current FPS value

**Implementation:**
- Uses rolling average for smooth FPS display
- Updates every frame for real-time monitoring

---

## Class Attributes

### Public Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `landmarker` | HandLandmarker | MediaPipe detector instance |
| `latest_result` | HandLandmarkerResult | Most recent detection result |
| `show_landmarks` | bool | Toggle landmark point visibility |
| `show_connections` | bool | Toggle connection line visibility |
| `is_paused` | bool | Pause state for detection |

### Private Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_fps_counter` | int | Frame counter for FPS calculation |
| `_fps_start_time` | float | Start time for FPS measurement |
| `_current_fps` | float | Current frames per second |

---

## Configuration Constants

### Detection Parameters

```python
# Confidence thresholds (0.0 - 1.0)
MIN_DETECTION_CONFIDENCE = 0.5    # Hand detection sensitivity
MIN_TRACKING_CONFIDENCE = 0.5     # Tracking stability
MIN_PRESENCE_CONFIDENCE = 0.5     # Hand presence detection

# Performance settings
MAX_NUM_HANDS = 2                 # Maximum hands to detect
RUNNING_MODE = LIVE_STREAM        # Real-time processing mode
```

### Visual Styling

```python
# Colors (BGR format)
LANDMARK_COLOR = (0, 255, 0)      # Green landmarks
CONNECTION_COLOR = (255, 0, 0)    # Red connections
TEXT_COLOR = (255, 255, 255)      # White text

# Sizes
LANDMARK_RADIUS = 3               # Landmark point size
CONNECTION_THICKNESS = 2          # Connection line thickness
TEXT_SCALE = 0.7                  # Text size scale
```

---

## Error Handling

### Exception Types

| Exception | Cause | Solution |
|-----------|-------|----------|
| `RuntimeError` | MediaPipe initialization failure | Check model file and dependencies |
| `FileNotFoundError` | Custom model not found | Verify model path exists |
| `cv2.error` | Camera access failure | Check camera permissions and availability |
| `ImportError` | Missing dependencies | Install required packages |

### Error Recovery

```python
try:
    detector = MediaPipeHandDetector()
    detector.run()
except RuntimeError as e:
    print(f"MediaPipe error: {e}")
    # Fallback to basic detection or retry
except cv2.error as e:
    print(f"Camera error: {e}")
    # Try different camera_id or check permissions
finally:
    detector.cleanup()
```

---

## Performance Optimization

### Best Practices

1. **Camera Resolution**: Use 720p for balance of quality and performance
2. **Lighting**: Ensure good lighting for better detection accuracy
3. **Background**: Use contrasting backgrounds for improved detection
4. **Distance**: Keep hands 0.5-3.0 meters from camera
5. **CPU Usage**: Close unnecessary applications for better performance

### Performance Metrics

| Metric | Typical Value | Optimal Range |
|--------|---------------|---------------|
| FPS | 30+ | 25-60 |
| CPU Usage | 15-30% | <50% |
| Memory | 200-400MB | <1GB |
| Latency | 30-50ms | <100ms |

---

## Integration Examples

### Basic Integration

```python
from mediapipe_hand_detector import MediaPipeHandDetector

def main():
    detector = MediaPipeHandDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        print("Detection stopped by user")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()
```

### Advanced Integration

```python
import cv2
from mediapipe_hand_detector import MediaPipeHandDetector

class CustomHandTracker(MediaPipeHandDetector):
    def __init__(self):
        super().__init__()
        self.gesture_history = []
    
    def _result_callback(self, result, output_image, timestamp_ms):
        super()._result_callback(result, output_image, timestamp_ms)
        
        # Custom gesture processing
        if result.hand_landmarks:
            self.process_gestures(result.hand_landmarks)
    
    def process_gestures(self, landmarks):
        # Custom gesture recognition logic
        pass

# Usage
tracker = CustomHandTracker()
tracker.run()
```

---

## Version Compatibility

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Python | 3.8 | 3.9+ |
| MediaPipe | 0.10.0 | Latest |
| OpenCV | 4.5.0 | 4.8.0+ |
| NumPy | 1.19.0 | 1.21.0+ |

---

## Support and Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive guides and tutorials
- **Examples**: Sample code and use cases
- **Community**: Discussion forums and support