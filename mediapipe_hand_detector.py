#!/usr/bin/env python3
"""
MediaPipe Hand Landmark Detection
Official MediaPipe implementation for real-time hand landmark detection
Based on: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from datetime import datetime
import os
import json

class MediaPipeHandDetector:
    def __init__(self, model_path=None):
        """
        Initialize MediaPipe Hand Landmarker
        
        Args:
            model_path (str): Path to the MediaPipe hand landmarker model file
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Application state
        self.paused = False
        self.show_landmarks = True
        self.show_connections = True
        
        # MediaPipe configuration
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Initialize landmarker (will be set up in setup_landmarker)
        self.landmarker = None
        self.latest_result = None
        self.result_timestamp = 0
        
        # Image dimensions and projection matrix for efficient processing
        self.image_width = 640
        self.image_height = 480
        self.projection_matrix = np.array([
            [2.0/self.image_width, 0, -1],
            [0, -2.0/self.image_height, 1],
            [0, 0, 1]
        ])
        
        # Hand landmark names for reference
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        print("🤖 MediaPipe Hand Landmarker Initialized")
        print("📊 Features:")
        print("   • Official MediaPipe 21-point detection")
        print("   • Real-time hand tracking")
        print("   • 3D landmark coordinates")
        print("   • Multi-hand detection")
        print("   • Professional accuracy")
        print("")
        print("🎮 Controls:")
        print("   SPACE - Pause/Resume")
        print("   S - Save screenshot")
        print("   C - Toggle connections")
        print("   L - Toggle landmarks")
        print("   Q/ESC - Quit")
        print("="*50)
    
    def result_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
        Callback function for live stream mode
        
        Args:
            result: HandLandmarkerResult containing detected landmarks
            output_image: Processed image
            timestamp_ms: Timestamp in milliseconds
        """
        self.latest_result = result
        self.result_timestamp = timestamp_ms
    
    def setup_landmarker(self, model_path=None):
        """
        Set up MediaPipe Hand Landmarker with live stream mode
        
        Args:
            model_path (str): Path to model file (optional, uses default if None)
        """
        try:
            # Download model if not exists
            default_model_path = 'hand_landmarker.task'
            if not os.path.exists(default_model_path):
                print("Downloading MediaPipe hand landmarker model...")
                import urllib.request
                model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(model_url, default_model_path)
                print("Model downloaded successfully!")
            
            # Configure base options
            if model_path and os.path.exists(model_path):
                base_options = self.BaseOptions(model_asset_path=model_path)
            else:
                base_options = self.BaseOptions(model_asset_path=default_model_path)
            
            # Create HandLandmarker options
            options = self.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=self.VisionRunningMode.LIVE_STREAM,
                num_hands=2,  # Detect up to 2 hands simultaneously
                min_hand_detection_confidence=0.5,  # Balanced detection threshold
                min_hand_presence_confidence=0.5,  # Balanced presence threshold
                min_tracking_confidence=0.5,  # Balanced tracking threshold
                result_callback=self.result_callback
            )
            
            # Configure model for faster inference
            base_options.delegate = 'gpu'  # Use GPU acceleration if available
            base_options.cpu_num_thread = 4  # Use multiple CPU threads
            
            # Set image dimensions for landmark projection
            self.image_width = 640
            self.image_height = 480
            
            # Configure for live video processing with proper dimensions
            options.running_mode = self.VisionRunningMode.LIVE_STREAM
            options.result_callback = self.result_callback
            
            # Set normalized dimensions for proper landmark scaling
            self.scale_x = self.image_width / 640.0  # Normalize to MediaPipe's default width
            self.scale_y = self.image_height / 480.0  # Normalize to MediaPipe's default height
            
            # Configure base options for optimal performance
            base_options = mp.tasks.BaseOptions(
                model_asset_path='hand_landmarker.task',
                delegate=mp.tasks.BaseOptions.Delegate.GPU  # Enable GPU acceleration
            )
            
            self.landmarker = self.HandLandmarker.create_from_options(options)
            print("✅ MediaPipe Hand Landmarker configured successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up MediaPipe Hand Landmarker: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame with MediaPipe Hand Landmarker
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with landmarks drawn
        """
        if self.landmarker is None:
            return frame
            
        # Convert BGR to RGB for MediaPipe (using faster conversion)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image with dimensions and process immediately
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        timestamp_ms = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, timestamp_ms)
        
        # Draw landmarks if available for smoother display
        if self.latest_result and self.latest_result.hand_landmarks:
            self.draw_landmarks(frame, self.latest_result)
        
        return frame
    
    def draw_landmarks(self, frame, result):
        """
        Draw hand landmarks and connections on the frame
        
        Args:
            frame: Frame to draw on
            result: HandLandmarkerResult containing landmarks
        """
        height, width = frame.shape[:2]
        
        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
            # Convert landmarks to pixel coordinates using scaling factors
            landmark_points = []
            for landmark in hand_landmarks:
                # Apply scaling factors to landmark coordinates
                x = int(landmark.x * 640 * self.scale_x)
                y = int(landmark.y * 480 * self.scale_y)
                landmark_points.append((x, y))
            
            # Draw connections
            if self.show_connections:
                connections = [
                    # Thumb
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    # Index finger
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    # Middle finger
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    # Ring finger
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    # Pinky
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    # Palm
                    (5, 9), (9, 13), (13, 17)
                ]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                        start_point = landmark_points[start_idx]
                        end_point = landmark_points[end_idx]
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            
            # Draw landmark points
            if self.show_landmarks:
                for i, (x, y) in enumerate(landmark_points):
                    # Different colors for different finger parts
                    if i == 0:  # Wrist
                        color = (255, 0, 0)  # Blue
                    elif i in [1, 2, 3, 4]:  # Thumb
                        color = (0, 255, 255)  # Yellow
                    elif i in [5, 6, 7, 8]:  # Index
                        color = (255, 0, 255)  # Magenta
                    elif i in [9, 10, 11, 12]:  # Middle
                        color = (0, 255, 0)  # Green
                    elif i in [13, 14, 15, 16]:  # Ring
                        color = (255, 255, 0)  # Cyan
                    else:  # Pinky
                        color = (128, 0, 128)  # Purple
                    
                    cv2.circle(frame, (x, y), 5, color, -1)
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
                    
                    # Draw landmark index
                    cv2.putText(frame, str(i), (x + 8, y - 8), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Display hand information
            if result.handedness:
                handedness = result.handedness[hand_idx]
                if handedness:
                    hand_label = handedness[0].category_name
                    confidence = handedness[0].score
                    
                    # Find wrist position for text placement
                    wrist_x, wrist_y = landmark_points[0]
                    text = f"{hand_label} ({confidence:.2f})"
                    cv2.putText(frame, text, (wrist_x - 50, wrist_y - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def calculate_fps(self):
        """
        Calculate and update FPS
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_info(self, frame):
        """
        Draw information overlay on frame
        
        Args:
            frame: Frame to draw on
        """
        # FPS counter
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hand count
        hand_count = len(self.latest_result.hand_landmarks) if self.latest_result and self.latest_result.hand_landmarks else 0
        hand_text = f"Hands Detected: {hand_count}"
        cv2.putText(frame, hand_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # MediaPipe status
        status_text = "MediaPipe Hand Landmarker"
        cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Pause indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Controls
        controls = [
            "SPACE: Pause | S: Save | C: Connections",
            "L: Landmarks | Q/ESC: Quit"
        ]
        
        for i, control in enumerate(controls):
            y_pos = frame.shape[0] - 40 + (i * 20)
            cv2.putText(frame, control, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def save_screenshot(self, frame):
        """
        Save current frame as screenshot
        
        Args:
            frame: Frame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mediapipe_hand_detection_{timestamp}.jpg"
        
        # Create output directory if it doesn't exist
        output_dir = "hand_detection_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"📸 Screenshot saved: {filepath}")
    
    def cleanup(self):
        """
        Clean up resources
        """
        if self.landmarker:
            self.landmarker.close()
        print("🧹 MediaPipe Hand Landmarker cleaned up")

def main():
    """
    Main application function
    """
    detector = MediaPipeHandDetector()
    
    # Setup MediaPipe landmarker
    if not detector.setup_landmarker():
        print("❌ Failed to initialize MediaPipe Hand Landmarker")
        return
    
    # Try different camera indices
    camera_index = 0
    max_attempts = 2
    cap = None
    
    while camera_index < max_attempts:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow backend
        if cap.isOpened() and cap.read()[0]:  # Test if we can actually read frames
            break
        print(f"Trying camera index {camera_index}...")
        camera_index += 1
        cap.release()
    
    if not cap or not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Set camera properties for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Higher FPS for smoother tracking
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer for lower latency
    
    print("🎥 Camera started. MediaPipe hand detection active...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            if not detector.paused:
                # Process frame with MediaPipe
                frame = detector.process_frame(frame)
                
                # Calculate FPS
                detector.calculate_fps()
            
            # Draw information overlay
            detector.draw_info(frame)
            
            # Display frame
            cv2.imshow('MediaPipe Hand Landmark Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                detector.paused = not detector.paused
                print(f"⏸️ {'Paused' if detector.paused else 'Resumed'}")
            elif key == ord('s'):  # 's'
                detector.save_screenshot(frame)
            elif key == ord('c'):  # 'c'
                detector.show_connections = not detector.show_connections
                print(f"🔗 Connections: {'ON' if detector.show_connections else 'OFF'}")
            elif key == ord('l'):  # 'l'
                detector.show_landmarks = not detector.show_landmarks
                print(f"📍 Landmarks: {'ON' if detector.show_landmarks else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("👋 MediaPipe Hand Detection stopped")

if __name__ == "__main__":
    main()