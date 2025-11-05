"""
Camera capture and frame handling utilities
"""
import cv2
import numpy as np
import subprocess
import json
import platform
from typing import Optional, Tuple, List, Dict
from PIL import Image


def get_available_cameras() -> List[Dict[str, any]]:
    """
    Get list of available cameras on macOS and other platforms.

    Returns:
        List of camera information dictionaries with 'id', 'name', and 'is_builtin' keys
    """
    cameras = []
    system = platform.system()

    if system == "Darwin":  # macOS
        cameras = _get_cameras_macos()
    else:
        cameras = _get_cameras_generic()

    return cameras


def _get_cameras_macos() -> List[Dict[str, any]]:
    """
    Get cameras on macOS using system_profiler command.
    Prioritizes built-in cameras.
    """
    cameras = []

    try:
        # Use system_profiler to get camera information
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            camera_info = data.get("SPCameraDataType", [])

            for i, camera in enumerate(camera_info):
                name = camera.get("_name", f"Camera {i}")
                # Check if it's built-in camera
                is_builtin = "FaceTime" in name or "built-in" in name.lower()

                cameras.append({
                    "id": i,
                    "name": name,
                    "is_builtin": is_builtin,
                    "model": camera.get("_model", "Unknown")
                })
    except Exception as e:
        print(f"Warning: Could not get camera info from system_profiler: {e}")

    # Fallback: test device IDs 0-5
    if not cameras:
        cameras = _get_cameras_generic(max_devices=6)

    return cameras


def _get_cameras_generic(max_devices: int = 10) -> List[Dict[str, any]]:
    """
    Get cameras by testing device IDs (works on all platforms).
    """
    cameras = []

    for i in range(max_devices):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name if available
                name = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cameras.append({
                    "id": i,
                    "name": f"Camera {i}",
                    "is_builtin": i == 0,  # Assume device 0 is built-in
                })
                cap.release()
        except Exception:
            pass

    return cameras


def get_builtin_camera_id() -> int:
    """
    Get the device ID of the built-in camera (usually 0 on macOS).
    Returns 0 if no built-in camera is found.
    """
    cameras = get_available_cameras()

    # First, try to find an explicitly marked built-in camera
    for camera in cameras:
        if camera.get("is_builtin", False):
            return camera["id"]

    # Fallback to device 0
    return 0


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
            print(f"Attempting to initialize camera with device_id: {self.device_id}")

            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"Error: Unable to open camera device {self.device_id}")

                # Try fallback to built-in camera
                if platform.system() == "Darwin":  # macOS
                    print("Trying built-in camera as fallback...")
                    builtin_id = get_builtin_camera_id()
                    if builtin_id != self.device_id:
                        print(f"Retrying with built-in camera (device_id: {builtin_id})")
                        self.device_id = builtin_id
                        self.cap = cv2.VideoCapture(builtin_id)
                        if not self.cap.isOpened():
                            print(f"Error: Unable to open built-in camera device {builtin_id}")
                            return False
                    else:
                        return False
                else:
                    return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Get actual properties (for verification)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"Camera initialized successfully (device_id: {self.device_id})")
            print(f"Requested resolution: {self.width}x{self.height} @ {self.fps}fps")
            print(f"Actual resolution:    {actual_width}x{actual_height} @ {actual_fps}fps")

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
