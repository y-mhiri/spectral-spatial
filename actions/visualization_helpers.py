import numpy as np
from PIL import Image
import os
from scipy.linalg import svd
from scipy.ndimage import zoom

def normalize_image(img_array, method='minmax'):
    """Normalize image array to [0,1] range"""
    if method == 'minmax':
        min_val, max_val = np.min(img_array), np.max(img_array)
        if max_val > min_val:
            return (img_array - min_val) / (max_val - min_val)
        return img_array
    elif method == 'percentile':
        p1, p99 = np.percentile(img_array, [1, 99])
        img_clipped = np.clip(img_array, p1, p99)
        return (img_clipped - p1) / (p99 - p1) if p99 > p1 else img_clipped
    return img_array

def resize_image(img_array, target_size=None, method='bicubic'):
    """Resize image with interpolation"""
    if target_size is None:
        return img_array
    
    if img_array.ndim == 2:
        h, w = img_array.shape
        if h != target_size or w != target_size:
            zoom_factor = target_size / max(h, w)
            return zoom(img_array, zoom_factor, order=3 if method == 'bicubic' else 1)
    
    elif img_array.ndim == 3:
        h, w, c = img_array.shape
        if h != target_size or w != target_size:
            zoom_factor = target_size / max(h, w)
            return zoom(img_array, (zoom_factor, zoom_factor, 1), order=3 if method == 'bicubic' else 1)
    
    return img_array

def get_eigenimages(hsi_data):
    """Compute eigenimages via SVD"""
    num_bands, height, width = hsi_data.shape
    hsi_reshaped = hsi_data.reshape(num_bands, -1)
    _, _, V = svd(hsi_reshaped, full_matrices=False)
    eigenimages = V.reshape(num_bands, height, width)
    return eigenimages

def extract_rgb_image(hsi_data, rgb_indices):
    """Extract RGB image from hyperspectral data"""
    if rgb_indices is None or len(rgb_indices) != 3:
        # Use first 3 bands as fallback
        rgb_indices = [0, 1, 2] if hsi_data.shape[0] >= 3 else [0, 0, 0]
    
    rgb_image = hsi_data[rgb_indices].transpose(1, 2, 0)  # [H, W, 3]
    return normalize_image(rgb_image)

def extract_eigenimage_rgb(hsi_data):
    """Extract RGB from first 3 eigenimages"""
    eigenimages = get_eigenimages(hsi_data)
    rgb_eigen = eigenimages[:3].transpose(1, 2, 0)  # [H, W, 3]
    return normalize_image(rgb_eigen)

def extract_band_image(hsi_data, band_idx):
    """Extract single band as grayscale"""
    if band_idx >= hsi_data.shape[0]:
        band_idx = 0
    band_image = hsi_data[band_idx]
    return normalize_image(band_image)

def save_image(img_array, filepath, target_size=None, dpi=300):
    """Save image array as PNG with high quality"""
    # Resize if needed
    if target_size is not None:
        img_array = resize_image(img_array, target_size)
    
    # Convert to uint8
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # Determine mode
    if img_array.ndim == 2:
        mode = 'L'
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        mode = 'RGB'
    else:
        raise ValueError("Image must be grayscale (2D) or RGB (3D)")
    
    # Create and save PIL image
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pil_image = Image.fromarray(img_uint8, mode=mode)
    pil_image.save(filepath, dpi=(dpi, dpi), optimize=True)

def create_comparison_grid(images, labels, target_size=None):
    """Create side-by-side comparison grid"""
    if not images or len(images) != len(labels):
        raise ValueError("Images and labels must have same length")
    
    # Resize all images to same size
    if target_size is not None:
        images = [resize_image(img, target_size) for img in images]
    
    # Ensure all images have same dimensions
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_h, max_w = max(heights), max(widths)
    
    # Pad images to same size
    padded_images = []
    for img in images:
        h, w = img.shape[:2]
        if img.ndim == 2:
            padded = np.zeros((max_h, max_w))
            padded[:h, :w] = img
        else:
            padded = np.zeros((max_h, max_w, img.shape[2]))
            padded[:h, :w] = img
        padded_images.append(padded)
    
    # Create horizontal concatenation
    if len(padded_images) == 1:
        return padded_images[0]
    
    grid = np.concatenate(padded_images, axis=1)
    return grid

def generate_filename(metadata, image_type, img_idx=0, extension='png'):
    """Generate descriptive filename from metadata"""
    # Extract key parameters
    algorithm = extract_parameter_value(metadata, 'algorithm', 'Unknown')
    lmbda = extract_parameter_value(metadata, 'lambda', 0)
    lmbda_m = extract_parameter_value(metadata, 'lambda_m', 0)
    noise = extract_parameter_value(metadata, 'noise_level', 0)
    
    # Get PSNR if available
    psnr_values = extract_metric_value(metadata, 'PSNR')
    psnr = np.mean(psnr_values) if psnr_values else 0
    
    # Format filename
    filename = f"{algorithm}_{lmbda:.0e}_{lmbda_m:.1f}_psnr{psnr:.1f}_noise{noise:.0f}_{image_type}_img{img_idx}.{extension}"
    return filename

def extract_parameter_value(metadata, param_name, default=None):
    """Extract parameter from metadata (simplified from action_helpers)"""
    if 'parameters' in metadata and param_name in metadata['parameters']:
        return metadata['parameters'][param_name]
    if 'zarr_metadata' in metadata and param_name in metadata['zarr_metadata']:
        return metadata['zarr_metadata'][param_name]
    
    # Handle variations
    variations = {'lambda': ['lmbda', 'lambda'], 'lambda_m': ['lmbda_m', 'lambda_m']}
    if param_name in variations:
        for var in variations[param_name]:
            if 'parameters' in metadata and var in metadata['parameters']:
                return metadata['parameters'][var]
            if 'zarr_metadata' in metadata and var in metadata['zarr_metadata']:
                return metadata['zarr_metadata'][var]
    
    return default

def extract_metric_value(metadata, metric_name):
    """Extract metric from metadata (simplified from action_helpers)"""
    if 'zarr_metadata' in metadata and metric_name in metadata['zarr_metadata']:
        return metadata['zarr_metadata'][metric_name]
    return None