"""
Conditioning tensor generation for neural rendering.

This module provides configuration and utilities for preparing conditioning
tensors from multi-sample input frames.
"""
from typing import Dict, List, Any, Tuple, Union, Optional
from enum import IntEnum

import torch
import numpy as np


# Configuration constants
SRC_CONFIG: Dict[str, Any] = {
    "sequences": 1024,
    "files": "sampleset_v1/train/scene{index:04d}.zip",
    "frames_per_sequence": 64,
    "val_frames": 2,
    "crop": 256,
    "samples": 32,
    "rendering_height": 1080,
    "rendering_width": 1920
}


TEST8_CONFIG: Dict[str, Any] = {
    "sequences": [
        {"name": "bistro1", "frames": 160},
        {"name": "bistro2", "frames": 160},
        {"name": "bistro3", "frames": 160},
        {"name": "bistro1-day", "frames": 160},
        {"name": "bistro2-day", "frames": 160},
        {"name": "measure7", "frames": 160},
        {"name": "bedroom", "frames": 80},
        {"name": "dining-room", "frames": 80},
        {"name": "kitchen", "frames": 80}
    ],
    "samples": 8,
    "rendering_height": 1080,
    "rendering_width": 1920,
    "files": "sampleset_v1/test8/{sequence_name}/frame{index:04d}.zip"
}


TEST32_CONFIG: Dict[str, Any] = {
    "sequences": [
        {"name": "bistro1", "frames": 40},
        {"name": "bistro2", "frames": 40},
        {"name": "bistro3", "frames": 40},
        {"name": "bistro1-day", "frames": 40},
        {"name": "bedroom", "frames": 40},
        {"name": "dining-room", "frames": 40},
        {"name": "kitchen", "frames": 40}
    ],
    "samples": 32,
    "rendering_height": 1080,
    "rendering_width": 1920,
    "files": "sampleset_v1/test32/{sequence_name}/frame{index:04d}.zip"
}


# Required buffer types
BUFFERS: List[str] = [
    "normal", "depth", "diffuse", "color", "reference"
]


# Default number of data samples
DATA_SAMPLES: int = 32


class TonemapperType(IntEnum):
    """Tonemapper types."""
    LOG10 = 0
    LOG_GAMMA = 1
    FILMIC = 2


def custom_sampler(min_samples: int, max_samples: int, B: int, device: str = 'cuda') -> torch.Tensor:
    """
    Create a custom sampler with higher probability for specific values.
    
    Args:
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples
        B: Batch size
        device: Computation device
        
    Returns:
        Tensor of sample counts for each item in the batch
    """
    # Input validation
    if min_samples < 1 or max_samples < min_samples:
        raise ValueError(f"Invalid sample range: min={min_samples}, max={max_samples}")
    
    # Create the full range of integers
    full_range = torch.arange(min_samples, max_samples + 1, device=device)
    
    # Special handling if range is too small
    range_size = max_samples - min_samples + 1
    if range_size <= 3:
        # Simple uniform sampling for small ranges
        return torch.randint(min_samples, max_samples + 1, size=(B,), device=device)
    
    # Create the probability distribution
    prob = torch.full((range_size,), 0.2 / (range_size - 3), device=device)
    
    # Set higher probability for [2, 4, 8, 32], we know SPP at test time
    special_values = torch.tensor([2, 4, 8, 32], device=device)
    
    # Filter valid special values that are within our range
    mask = (special_values >= min_samples) & (special_values <= max_samples)
    if mask.any():
        valid_special_values = special_values[mask]
        special_indices = valid_special_values - min_samples
        prob[special_indices] = 0.8 / len(valid_special_values)
    
    # Normalize probabilities
    prob /= prob.sum()
    
    # Sample from the custom distribution
    return full_range[torch.multinomial(prob, B, replacement=True)]


def _tonemap(x: Union[np.ndarray, torch.Tensor], 
            tonemapper: int = TonemapperType.LOG10) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply tonemapping to HDR content.
    
    Args:
        x: Input tensor or array
        tonemapper: Tonemapping method to use
        
    Returns:
        Tonemapped tensor or array
    """
    if tonemapper == TonemapperType.LOG10:
        return torch.log10(1 + x) if torch.is_tensor(x) else np.log10(1 + x)
    elif tonemapper == TonemapperType.LOG_GAMMA:
        return torch.log(1 + x) / 2.2 if torch.is_tensor(x) else np.log(1 + x) / 2.2
    elif tonemapper == TonemapperType.FILMIC:
        return 0.47 * torch.pow(x, 1/2.4) if torch.is_tensor(x) else 0.47 * np.power(x, 1/2.4)
    else:
        raise ValueError(f"Invalid tonemapper value {tonemapper}. Must be 0, 1, or 2.")


def _undoTonemap(x: Union[np.ndarray, torch.Tensor], 
                tonemapper: int = TonemapperType.LOG10) -> Union[np.ndarray, torch.Tensor]:
    """
    Reverse tonemapping operation.
    
    Args:
        x: Input tensor or array
        tonemapper: Tonemapping method to reverse
        
    Returns:
        Original HDR tensor or array
    """
    if tonemapper == TonemapperType.LOG10:
        return 10**x - 1
    elif tonemapper == TonemapperType.LOG_GAMMA:
        return torch.exp(2.2*x) - 1 if torch.is_tensor(x) else np.exp(2.2*x) - 1
    elif tonemapper == TonemapperType.FILMIC:
        return torch.pow(x / 0.47, 2.4) if torch.is_tensor(x) else np.power(x / 0.47, 2.4)
    else:
        raise ValueError(f"Invalid tonemapper value {tonemapper}. Must be 0, 1, or 2.")


def _tm(inp: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply filmic tonemapping (shorthand).
    
    Args:
        inp: Input tensor or array
        
    Returns:
        Tonemapped tensor or array
    """
    return 0.47 * torch.pow(inp, 1/2.4) if torch.is_tensor(inp) else 0.47 * np.power(inp, 1/2.4)


def un_tm(inp: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Reverse filmic tonemapping (shorthand).
    
    Args:
        inp: Input tensor or array
        
    Returns:
        Original HDR tensor or array
    """
    return torch.pow(inp/0.47, 2.4) if torch.is_tensor(inp) else np.power(inp/0.47, 2.4)


def get_conditioning(
    frame: Dict[str, torch.Tensor],
    min_samples: int = 1,
    max_samples: int = 32,
    tonemapper: int = TonemapperType.LOG_GAMMA
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate conditioning tensors from input frame buffers.
    
    Args:
        frame: Dictionary of input buffers
        min_samples: Minimum number of samples to use
        max_samples: Maximum number of samples to use
        tonemapper: Tonemapping method
        
    Returns:
        Tuple of (color_tensor, conditioning_tensor, reference_tensor)
    """
    # Validate input parameters
    assert 1 <= min_samples <= max_samples, f"Invalid sample range: min={min_samples}, max={max_samples}"
    
    # Move reference to device and get dimensions
    reference = frame['reference'].cuda() 
    B, C, H, W = reference.shape

    def process_full_samples() -> Dict[str, torch.Tensor]:
        """Process using all available samples."""
        return {
            'color': frame['color'].cuda().mean(-1), 
            'diffuse': frame['diffuse'].cuda().mean(-1).clamp(0, 1),
            'depth': frame['depth'].cuda().mean(-1).clamp(0, 1),
            'normal': frame['normal'].cuda().mean(-1),
            'count': torch.ones(B, 1, H, W, device='cuda') * min_samples / DATA_SAMPLES,
        }

    def process_partial_samples() -> Dict[str, torch.Tensor]:
        """Process using a random subset of samples."""
        # Determine sample count per batch item
        if min_samples == max_samples:
            mat = torch.full((B,), min_samples, device='cuda')
        else:
            mat = custom_sampler(min_samples, max_samples, B)

        # Create random sample selection mask
        random_values = torch.rand(B, DATA_SAMPLES, device='cuda')
        shuffled_indices = torch.argsort(random_values, dim=1)
        
        range_tensor = torch.arange(DATA_SAMPLES, device='cuda').expand(B, -1)
        mask = range_tensor < mat.unsqueeze(1)
        mask = mask.gather(1, shuffled_indices)
        
        # Expand mask for tensor operations
        mask = mask.float().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, 1, H, W, -1)
        count = mat.view(B, 1, 1, 1)

        def masked_mean(inp: torch.Tensor) -> torch.Tensor:
            """Compute masked mean across sample dimension."""
            return (inp.cuda() * mask).sum(dim=-1) / count

        # Apply masked mean to each buffer
        return {
            'color': masked_mean(frame['color']),
            'diffuse': masked_mean(frame['diffuse']).clamp(0, 1),
            'depth': masked_mean(frame['depth']).clamp(0, 1),
            'normal': masked_mean(frame['normal']),
            'count': (mask.sum(dim=-1) / DATA_SAMPLES),
        }

    # Choose processing method based on sample count
    use_full_samples = (min_samples == frame['color'].shape[-1] and min_samples == max_samples)
    results = process_full_samples() if use_full_samples else process_partial_samples()

    # Normalize normal vectors and remap to [0, 1] range
    normal_magnitude = torch.norm(results['normal'], dim=1, keepdim=True).clamp(min=1e-8)
    results['normal'] = results['normal'] / normal_magnitude
    results['normal'] = (0.5 * (results['normal'] + 1)).clamp(0, 1)

    # Apply tonemapping
    reference = _tonemap(reference, tonemapper).clamp(0, 1)
    results['color'] = _tonemap(results['color'], tonemapper).clamp(0, 1)

    # Concatenate features for conditioning tensor
    conditioning = torch.cat([
        results['color'], 
        results['diffuse'], 
        results['normal'], 
        results['depth'], 
        results['count']
    ], dim=1)

    return results['color'], conditioning, reference