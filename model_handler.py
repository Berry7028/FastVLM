"""
FastVLM model handler for image description generation
"""
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set custom Hugging Face cache directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(MODELS_DIR)

# FastVLM specific constants
IMAGE_TOKEN_INDEX = -200


class FastVLMHandler:
    """Handle FastVLM model initialization and inference"""

    def __init__(self, model_name: str = "apple/FastVLM-1.5B", device: str = "cuda"):
        """
        Initialize FastVLM model handler

        Args:
            model_name: Model identifier on HuggingFace
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        logger.info(f"FastVLMHandler initialized for {self.model_name} on {self.device}")

    def load_model(self) -> bool:
        """Load FastVLM model from HuggingFace"""
        try:
            logger.info(f"Loading model: {self.model_name}")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )

            # Load model with appropriate precision
            logger.info("Loading model weights...")
            try:
                # Try with flash_attention_2 first (if CUDA available)
                if self.device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2"
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True
                    )
            except Exception as e:
                # Fallback without flash_attention_2
                logger.warning(f"Flash attention not available, using standard attention: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else "cpu",
                    trust_remote_code=True
                )

            self.model.eval()
            self._is_loaded = True
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def generate_description(
        self,
        image: Image.Image,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Generate description for the given image

        Args:
            image: PIL Image object
            max_length: Maximum length of generated text
            temperature: Sampling temperature

        Returns:
            Generated description or None if inference fails
        """
        if not self._is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return None

        try:
            # Prepare image for the model
            # FastVLM expects the image to be processed by its internal vision encoder
            image = image.convert("RGB")

            # Build the prompt with image token placeholder
            # The image token index is -200 for FastVLM
            prompt = f"<image>{IMAGE_TOKEN_INDEX}</image>\nDescribe this image:"

            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            )

            # Move inputs to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Prepare image for the model
            # The model will extract vision features internally
            # We need to pass pixel values through the vision encoder
            image_tensor = self._prepare_image(image)

            # Generate description
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor if hasattr(self.model, 'vision_tower') else None,
                    pixel_values=image_tensor,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )

            # Decode output
            description = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return description.strip()

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prepare image tensor for the model

        Args:
            image: PIL Image object

        Returns:
            Image tensor
        """
        try:
            # Resize image to a standard size
            image = image.resize((336, 336))

            # Convert to tensor
            import numpy as np
            image_array = np.array(image).astype(np.float32) / 255.0

            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std

            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            return torch.zeros(1, 3, 336, 336).to(self.device)

    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self._is_loaded and self.model is not None

    def cleanup(self):
        """Cleanup and free resources"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache() if self.device == "cuda" else None
        logger.info("Resources cleaned up")
