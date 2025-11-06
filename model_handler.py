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
import numpy as np

logger = logging.getLogger(__name__)

# Set custom Hugging Face cache directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(MODELS_DIR)

# FastVLM specific constants
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def tokenizer_image_token(
    prompt: str,
    tokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: Optional[str] = None
) -> torch.Tensor:
    """
    Tokenize prompt with special handling for image tokens.
    Replaces <image> placeholders with special token indices.

    Args:
        prompt: Text prompt containing <image> token(s)
        tokenizer: Tokenizer to use
        image_token_index: Special token index for images (default -200)
        return_tensors: Format for output ("pt" for PyTorch tensor)

    Returns:
        Tokenized input_ids as tensor
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0

    # Handle BOS token
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # Insert image tokens between text chunks
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)

    return input_ids


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

            if self.device == "cuda":
                try:
                    # Try with flash_attention_2 first (if CUDA available)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2"
                    ).to(self.device)
                except Exception as e:
                    # Fallback without flash_attention_2
                    logger.warning(f"Flash attention not available: {e}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    ).to(self.device)
            else:
                # CPU loading - no device_map, just use .to()
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)

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
            image = image.convert("RGB")

            # FastVLM requires <image> token in the prompt
            prompt = f"{DEFAULT_IMAGE_TOKEN}\nDescribe this image in detail."

            # Prepare input_ids with special image token handling
            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            )
            input_ids = input_ids.unsqueeze(0).to(self.device)

            # Process image through the model's image processor
            image_tensor = self._prepare_image_for_model(image)

            # Generate description using the correct API
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    max_new_tokens=max_length,
                    use_cache=True
                )

            # Decode output
            description = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip()

            return description

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _prepare_image_for_model(self, image: Image.Image) -> torch.Tensor:
        """
        Prepare image tensor for FastVLM model using proper image processor

        Args:
            image: PIL Image object

        Returns:
            Image tensor in the format expected by FastVLM
        """
        try:
            # Try to get the image processor from the model's vision tower
            if hasattr(self.model, 'get_vision_tower'):
                vision_tower = self.model.get_vision_tower()
                if hasattr(vision_tower, 'image_processor'):
                    image_processor = vision_tower.image_processor
                    processed = image_processor(images=image, return_tensors="pt")
                    pixel_values = processed.get("pixel_values")
                    if pixel_values is not None:
                        return pixel_values.to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)

            # Fallback: use model's image_processor attribute if available
            if hasattr(self.model, 'image_processor'):
                processed = self.model.image_processor(images=image, return_tensors="pt")
                pixel_values = processed.get("pixel_values")
                if pixel_values is not None:
                    return pixel_values.to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)

            # Final fallback: manual image processing
            logger.warning("Using fallback image processing method")
            image = image.resize((336, 336))
            image_array = np.array(image).astype(np.float32) / 255.0

            # Normalize with ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std

            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            return image_tensor.to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)

        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a dummy tensor with appropriate dtype
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            return torch.zeros(1, 3, 336, 336, dtype=dtype).to(self.device)

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prepare image tensor for the model (deprecated - use _prepare_image_for_model)

        Args:
            image: PIL Image object

        Returns:
            Image tensor
        """
        return self._prepare_image_for_model(image)

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
