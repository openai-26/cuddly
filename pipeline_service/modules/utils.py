from PIL import Image
from PIL import ImageStat

import io
import base64
from datetime import datetime
from typing import Optional, Tuple
import os
import random
import numpy as np
import torch

from logger_config import logger
from schemas.trellis_schemas import TrellisResult

from config import settings

def secure_randint(low: int, high: int) -> int:
    """ Return a random integer in [low, high] using os.urandom. """
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)

def set_random_seed(seed: int) -> None:
    """ Function for setting global seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decode_image(prompt: str) -> Image.Image:
    """
    Decode the image from the base64 string.

    Args:
        prompt: The base64 string of the image.

    Returns:
        The image.
    """
    # Decode the image from the base64 string
    image_bytes = base64.b64decode(prompt)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def to_png_base64(image: Image.Image) -> str:
    """
    Convert the image to PNG format and encode to base64.

    Args:
        image: The image to convert.

    Returns:
        Base64 encoded PNG image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # Convert to base64 from bytes to string
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_file_bytes(data: bytes, folder: str, prefix: str, suffix: str) -> None:
    """
    Save binary data to the output directory.

    Args:
        data: The data to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        suffix: The suffix of the file.
    """
    target_dir = settings.output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{prefix}_{timestamp}{suffix}"
    try:
        path.write_bytes(data)
        logger.debug(f"Saved file {path}")
    except Exception as exc:
        logger.error(f"Failed to save file {path}: {exc}")

def save_image(image: Image.Image, folder: str, prefix: str, timestamp: str) -> None:
    """
    Save PIL Image to the output directory.

    Args:
        image: The PIL Image to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        timestamp: The timestamp of the file.
    """
    target_dir = settings.output_dir / folder / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{prefix}.png"
    try:
        image.save(path, format="PNG")
        logger.debug(f"Saved image {path}")
    except Exception as exc:
        logger.error(f"Failed to save image {path}: {exc}")

def save_files(
    trellis_result: Optional[TrellisResult], 
    image_edited: Image.Image, 
    image_without_background: Image.Image
) -> None:
    """
    Save the generated files to the output directory.

    Args:
        trellis_result: The Trellis result to save.
        image_edited: The edited image to save.
        image_without_background: The image without background to save.
    """
    # Save the Trellis result if available
    if trellis_result:
        if trellis_result.ply_file:
            save_file_bytes(trellis_result.ply_file, "ply", "mesh", suffix=".ply")

    # Save the images using PIL Image.save()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_image(image_edited, "png", "image_edited", timestamp)
    save_image(image_without_background, "png", "image_without_background", timestamp)


def validate_image_quality(image: Image.Image, reference_image: Optional[Image.Image] = None) -> Tuple[bool, dict]:
    """
    Validate image quality by checking for noise, artifacts, and other issues.
    
    Args:
        image: The image to validate
        reference_image: Optional reference image to compare against
        
    Returns:
        Tuple of (is_valid, quality_metrics) where:
        - is_valid: True if image passes quality checks
        - quality_metrics: Dictionary with quality metrics
    """
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image.convert("RGB"))
        
        # Calculate basic statistics
        stats = ImageStat.Stat(image)
        mean_brightness = sum(stats.mean) / len(stats.mean)
        std_dev = sum(stats.stddev) / len(stats.stddev)
        
        # Calculate variance (high variance can indicate noise)
        variance = np.var(img_array)
        
        # Calculate Laplacian variance (detects blur/noise)
        # Convert to grayscale for Laplacian
        gray = np.mean(img_array, axis=2).astype(np.float32)
        
        # Simple Laplacian-like variance calculation
        # Calculate variance of differences between adjacent pixels
        if gray.size > 1:
            h_diff = np.diff(gray, axis=0)
            w_diff = np.diff(gray, axis=1)
            laplacian_var = float(np.var(h_diff) + np.var(w_diff))
        else:
            laplacian_var = 0.0
        
        # Check for extreme values (potential artifacts)
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        extreme_pixels = np.sum((img_array < 5) | (img_array > 250))
        extreme_ratio = extreme_pixels / img_array.size
        
        # Check for uniform regions (potential corruption)
        # Calculate local variance in small patches
        patch_size = 8
        h, w = gray.shape
        local_vars = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                local_vars.append(np.var(patch))
        
        avg_local_var = np.mean(local_vars) if local_vars else 0
        
        # Quality thresholds
        quality_metrics = {
            "mean_brightness": mean_brightness,
            "std_dev": std_dev,
            "variance": float(variance),
            "laplacian_variance": float(laplacian_var),
            "extreme_pixel_ratio": extreme_ratio,
            "avg_local_variance": float(avg_local_var),
        }
        
        # Validation criteria
        is_valid = True
        issues = []
        
        # Check for excessive noise (very high variance relative to mean)
        if variance > 10000:  # Threshold for excessive noise
            is_valid = False
            issues.append("excessive_variance")
        
        # Check for too many extreme pixels (potential artifacts)
        if extreme_ratio > 0.1:  # More than 10% extreme pixels
            is_valid = False
            issues.append("excessive_extreme_pixels")
        
        # Check for too uniform (potential corruption)
        if avg_local_var < 10:  # Very low local variance
            is_valid = False
            issues.append("too_uniform")
        
        # Check for reasonable brightness range
        if mean_brightness < 10 or mean_brightness > 245:
            is_valid = False
            issues.append("extreme_brightness")
        
        quality_metrics["is_valid"] = is_valid
        quality_metrics["issues"] = issues
        
        return is_valid, quality_metrics
        
    except Exception as e:
        logger.warning(f"Error validating image quality: {e}")
        # On error, assume valid but log the issue
        return True, {"error": str(e), "is_valid": True}


def preprocess_input_image(image: Image.Image) -> Image.Image:
    """
    Preprocess input image to improve editing quality.
    
    Args:
        image: Input image to preprocess
        
    Returns:
        Preprocessed image
    """
    try:
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get image statistics
        stats = ImageStat.Stat(image)
        mean_brightness = sum(stats.mean) / len(stats.mean)
        
        # Normalize brightness if too dark or too bright
        if mean_brightness < 30:
            # Too dark - apply slight brightening
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
            logger.debug("Applied brightness enhancement to dark image")
        elif mean_brightness > 225:
            # Too bright - apply slight darkening
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.9)
            logger.debug("Applied brightness reduction to bright image")
        
        return image
        
    except Exception as e:
        logger.warning(f"Error preprocessing image: {e}, returning original")
        return image.convert("RGB") if image.mode != "RGB" else image

