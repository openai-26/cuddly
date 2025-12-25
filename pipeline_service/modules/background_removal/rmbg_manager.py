from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop

from config import Settings
from logger_config import logger


class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService.
        """
        self.settings = settings

        # Set padding percentage, output size
        self.padding_percentage = self.settings.padding_percentage
        self.output_size = self.settings.output_image_size
        self.limit_padding = self.settings.limit_padding
        self.mask_threshold = self.settings.background_mask_threshold

        # Enhanced settings for improved preprocessing
        self.use_multi_threshold = True  # Use multiple thresholds for better mask quality
        self.use_adaptive_padding = True  # Use adaptive padding based on object shape
        self.min_mask_threshold = 0.7  # Lower threshold for better edge coverage
        self.max_mask_threshold = 0.9  # Higher threshold for core mask

        # Set device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # Set model
        self.model: AutoModelForImageSegmentation | None = None

        # Set transform
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.background_removal_model_id} model...")

        # Load model
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.settings.background_removal_model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            logger.success(f"{self.settings.background_removal_model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.background_removal_model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.background_removal_model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.background_removal_model_id} model not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        """
        try:
            t1 = time.time()
            # Check if the image has alpha channel
            has_alpha = False
            
            if image.mode == "RGBA":
                # Get alpha channel
                alpha = np.array(image)[:, :, 3]
                if not np.all(alpha==255):
                    has_alpha=True
            
            if has_alpha:
                # If the image has alpha channel, return the image
                output = image
                
            else:
                # PIL.Image (H, W, C) C=3
                rgb_image = image.convert('RGB')
                
                # Tensor (H, W, C) -> (C, H',W')
                rgb_tensor = self.transforms(rgb_image).to(self.device)
                output = self._remove_background(rgb_tensor)

                image_without_background = to_pil_image(output[:3])

            removal_time = time.time() - t1
            logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {image_without_background.size} - InputSize: {image.size}")

            return image_without_background
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return image 

    def _create_combined_mask(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Create mask using multiple thresholds for better edge quality.
        
        Args:
            preds: Model predictions (H, W) tensor with values in [0, 1]
            
        Returns:
            Combined mask (H, W) tensor
        """
        if not self.use_multi_threshold:
            # Fallback to single threshold
            return (preds > 0.8).float()
        
        # Use multiple thresholds and combine for smoother edges
        thresholds = [self.min_mask_threshold, 0.8, self.max_mask_threshold]
        weights = [0.2, 0.5, 0.3]  # Weight higher threshold more
        
        masks = []
        for threshold in thresholds:
            mask = (preds > threshold).float()
            masks.append(mask)
        
        # Weighted combination
        combined = sum(m * w for m, w in zip(masks, weights))
        
        # Apply morphological operations to clean up mask
        combined = combined.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        # Opening: remove small noise
        combined = F.max_pool2d(combined, kernel_size=3, stride=1, padding=1)
        # Closing: fill small holes
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
        combined = combined.squeeze(0).squeeze(0)
        
        return combined

    def _calculate_adaptive_padding(self, width: int, height: int, mask_shape: tuple) -> tuple[float, float]:
        """
        Calculate adaptive padding based on object characteristics.
        
        Args:
            width: Object width
            height: Object height
            mask_shape: Shape of the mask (H, W)
            
        Returns:
            Tuple of (padding_x_factor, padding_y_factor)
        """
        if not self.use_adaptive_padding or height == 0:
            return (self.padding_percentage, self.padding_percentage)
        
        aspect_ratio = width / height
        base_padding = self.padding_percentage
        
        # Adjust for aspect ratio
        if aspect_ratio > 1.5:  # Wide object (landscape)
            padding_x = base_padding * 1.3
            padding_y = base_padding * 0.8
        elif aspect_ratio < 0.67:  # Tall object (portrait)
            padding_x = base_padding * 0.8
            padding_y = base_padding * 1.3
        else:  # Square-ish
            padding_x = padding_y = base_padding
        
        # Adjust for size (smaller objects need more padding)
        mask_area = mask_shape[0] * mask_shape[1]
        object_area = width * height
        size_factor = object_area / mask_area if mask_area > 0 else 1.0
        
        if size_factor < 0.1:  # Very small object
            padding_x *= 1.5
            padding_y *= 1.5
        elif size_factor < 0.2:  # Small object
            padding_x *= 1.2
            padding_y *= 1.2
        
        return (padding_x, padding_y)

    def _remove_background(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image with improved processing.
        """
        # Normalize tensor value for background removal model, reshape for model batch processing (C=3, H, W) -> (1, C=3, H, W)
        input_tensor = self.normalize(image_tensor).unsqueeze(0)
                
        with torch.no_grad():
            # Get mask from model (1, 1, H, W)
            preds = self.model(input_tensor)[-1].sigmoid()
            # Reshape: (1, 1, H, W) -> (H, W)
            preds = preds[0].squeeze()
            
            # Create combined mask using multiple thresholds
            mask = self._create_combined_mask(preds)

        # Get bounding box indices using lower threshold for better coverage
        bbox_indices = torch.argwhere(mask > self.min_mask_threshold)
        if len(bbox_indices) == 0:
            # Fallback: use entire image
            crop_args = dict(top=0, left=0, height=mask.shape[1], width=mask.shape[0])
            padding_x = padding_y = self.padding_percentage
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            
            # Calculate adaptive padding
            padding_x, padding_y = self._calculate_adaptive_padding(width, height, mask.shape)
            
            center = (h_max + h_min) / 2, (w_max + w_min) / 2
            
            # Apply adaptive padding
            padded_width = int(width * (1 + padding_x))
            padded_height = int(height * (1 + padding_y))
            size = max(padded_width, padded_height)  # Keep square for compatibility
            
            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)

            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left
            )

        # Apply mask to image
        mask = mask.unsqueeze(0)
        # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
        tensor_rgba = torch.cat([image_tensor * mask, mask], dim=-3)
        
        # Crop and resize with antialiasing enabled for better quality
        output = resized_crop(tensor_rgba, **crop_args, size=self.output_size, antialias=True)
        return output

