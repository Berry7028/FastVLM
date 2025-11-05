"""
Real-time camera with FastVLM image description
Main application entry point
"""
import cv2
import yaml
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Set Hugging Face cache directory before importing model_handler
models_dir = Path(__file__).parent / "models"
models_dir.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(models_dir)

from camera_utils import CameraCapture
from model_handler import FastVLMHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraDescriptionApp:
    """Real-time camera application with FastVLM descriptions"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize application with configuration"""
        self.config = self._load_config(config_path)
        self.camera = None
        self.model_handler = None
        self.current_description = "Loading model..."
        self.frame_counter = 0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)

    def initialize(self) -> bool:
        """Initialize camera and model"""
        # Initialize camera
        camera_config = self.config.get('camera', {})
        self.camera = CameraCapture(
            device_id=camera_config.get('device_id', 0),
            width=camera_config.get('frame_width', 640),
            height=camera_config.get('frame_height', 480),
            fps=camera_config.get('fps', 30)
        )

        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False

        # Initialize model
        model_config = self.config.get('model', {})
        self.model_handler = FastVLMHandler(
            model_name=model_config.get('model_name', 'apple/FastVLM-1.5B'),
            device=model_config.get('device', 'cuda')
        )

        if not self.model_handler.load_model():
            logger.error("Failed to load model")
            return False

        logger.info("Application initialized successfully")
        return True

    def _draw_text_with_background(
        self,
        frame: Any,
        text: str,
        position: tuple,
        font_scale: float = 0.6,
        thickness: int = 2,
        text_color: tuple = (0, 255, 0),
        bg_color: tuple = (0, 0, 0)
    ) -> Any:
        """Draw text with background on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        x, y = position
        margin = 5

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - margin, y - text_size[1] - margin),
            (x + text_size[0] + margin, y + margin),
            bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness
        )

        return frame

    def process_frame(self, frame: Any) -> None:
        """Process frame and update description if needed"""
        self.frame_counter += 1
        processing_config = self.config.get('processing', {})
        inference_interval = processing_config.get('inference_interval', 2)

        # Process every N frames
        if self.frame_counter % inference_interval == 0:
            try:
                # Convert frame to PIL Image
                pil_image = self.camera.frame_to_pil(frame)

                # Generate description
                model_config = self.config.get('model', {})
                description = self.model_handler.generate_description(
                    image=pil_image,
                    max_length=model_config.get('max_length', 100),
                    temperature=model_config.get('temperature', 0.7)
                )

                if description:
                    self.current_description = description
                    logger.info(f"Description: {description}")

            except Exception as e:
                logger.error(f"Error processing frame: {e}")

    def draw_ui(self, frame: Any) -> Any:
        """Draw UI elements on frame"""
        processing_config = self.config.get('processing', {})
        font_size = processing_config.get('display_font_size', 0.6)
        text_color = tuple(processing_config.get('text_color', [0, 255, 0]))

        # Draw frame counter
        frame = self._draw_text_with_background(
            frame,
            f"Frame: {self.frame_counter}",
            (10, 30),
            font_scale=font_size,
            text_color=text_color
        )

        # Draw description (wrapped)
        max_chars_per_line = 60
        description_lines = [
            self.current_description[i:i+max_chars_per_line]
            for i in range(0, len(self.current_description), max_chars_per_line)
        ]

        y_offset = 70
        for line in description_lines[:3]:  # Show max 3 lines
            frame = self._draw_text_with_background(
                frame,
                line,
                (10, y_offset),
                font_scale=font_size,
                text_color=text_color
            )
            y_offset += 30

        # Draw controls info
        frame = self._draw_text_with_background(
            frame,
            "Press 'q' to quit | 's' to save screenshot",
            (10, frame.shape[0] - 20),
            font_scale=font_size * 0.8,
            text_color=text_color
        )

        return frame

    def run(self) -> None:
        """Run the main application loop"""
        if not self.initialize():
            logger.error("Failed to initialize application")
            return

        logger.info("Application started. Press 'q' to quit, 's' to save screenshot")

        try:
            while True:
                # Capture frame
                success, frame = self.camera.get_frame()
                if not success:
                    logger.warning("Failed to capture frame")
                    continue

                # Process frame
                self.process_frame(frame)

                # Draw UI
                frame = self.draw_ui(frame)

                # Display frame
                cv2.imshow("FastVLM Camera Description", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{self.frame_counter}.png"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.camera:
            self.camera.release()
        if self.model_handler:
            self.model_handler.cleanup()
        cv2.destroyAllWindows()
        logger.info("Application closed")


def main():
    """Main entry point"""
    app = CameraDescriptionApp()
    app.run()


if __name__ == "__main__":
    main()
