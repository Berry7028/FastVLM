"""
Camera capture and frame handling utilities
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image


class CameraCapture:
    """Handle camera capture and frame processing"""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize camera capture

        Args:
            device_id: Camera device ID (default: 0)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame_count = 0

    def initialize(self) -> bool:
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"Error: Unable to open camera device {self.device_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            print(f"Camera initialized successfully")
            print(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a frame from camera

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None

        try:
            success, frame = self.cap.read()
            if success:
                self.frame_count += 1
            return success, frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return False, None

    def frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert OpenCV frame to PIL Image"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def release(self):
        """Release camera resource"""
        if self.cap is not None:
            self.cap.release()
            print(f"Camera released. Total frames captured: {self.frame_count}")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
