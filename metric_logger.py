"""
Image quality metrics calculation and logging module.

This module provides a MetricLogger class for calculating and tracking various
image quality metrics, including standard metrics (PSNR, SSIM, L1), perceptual
metrics (FLIP, LPIPS), and deep learning-based similarity metrics (DINO, CLIP).
"""
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image
import pyfvvdp
import clip
from torchvision import transforms
from piq import psnr, ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import AutoImageProcessor, AutoModel

from flip_loss import LDRFLIPLoss, HDRFLIPLoss
from loss import L1Loss, relativeL1Loss, relativeL2Loss, SMAPE, tonemappedRelativeMSE


def _tonemap(img: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Apply tonemapping to HDR images.
    
    Args:
        img: Input HDR image tensor
        gamma: Gamma value for tonemapping
        
    Returns:
        Tonemapped LDR image tensor
    """
    img = torch.clamp(img, min=0.0, max=None)
    img = (img / (1 + img)) ** (1.0 / gamma)
    return img


@dataclass
class MetricResults:
    """Container for image quality metrics results."""
    # Standard metrics
    L1: float = 0.0
    Relative_L1: float = 0.0
    Relative_MSE: float = 0.0
    Tonemapped_Relative_MSE: float = 0.0
    SMAPE: float = 0.0
    PSNR: float = 0.0
    SSIM: float = 0.0
    
    # Perceptual metrics
    LPIPS: float = 0.0
    LDR_FLIP: float = 0.0
    HDR_FLIP: float = 0.0
    FOV: float = 0.0
    
    # Neural network-based metrics
    DINO: float = 0.0
    CLIP: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert results to dictionary with standardized keys."""
        return {
            # Convert attribute names to the expected output format
            "L1": self.L1,
            "Relative L1": self.Relative_L1,
            "Relative MSE": self.Relative_MSE,
            "Tonemapped Relative MSE": self.Tonemapped_Relative_MSE,
            "SMAPE": self.SMAPE,
            "PSNR": self.PSNR,
            "SSIM": self.SSIM,
            "LPIPS": self.LPIPS,
            "LDR_FLIP": self.LDR_FLIP,
            "HDR_FLIP": self.HDR_FLIP,
            "FOV": self.FOV,
            "DINO": self.DINO,
            "CLIP": self.CLIP,
        }


class MetricLogger:
    """
    Logger for tracking and computing image quality metrics.
    
    This class computes a wide range of metrics for comparing predicted images
    with ground truth, including traditional metrics like PSNR and SSIM as well
    as perceptual metrics like FLIP and neural network-based similarity metrics.
    
    Attributes:
        gamma: Gamma value for tonemapping HDR images to LDR
        total: Accumulated metric values
        total_bs: Total number of samples processed
    """
    
    def __init__(self, gamma: float = 2.4):
        """
        Initialize the MetricLogger.
        
        Args:
            gamma: Gamma value for tonemapping
        """
        self.gamma = gamma
        
        # Initialize model-based metrics
        self._init_models()
        
        # Initialize accumulators
        self.total = {
            "L1": 0.0,
            "Relative L1": 0.0,
            "Relative MSE": 0.0,
            "Tonemapped Relative MSE": 0.0,
            "SMAPE": 0.0,
            "PSNR": 0.0,
            "SSIM": 0.0,
            "LPIPS": 0.0,
            "DINO": 0.0,
            "CLIP": 0.0,
            "LDR_FLIP": 0.0,
            "HDR_FLIP": 0.0,
            "FOV": 0.0,
        }
        
        self.total_bs = 0

    def _init_models(self) -> None:
        """Initialize all models used for metric calculation."""
        # Initialize FLIP models
        self.hdr_flip = HDRFLIPLoss()
        self.ldr_flip = LDRFLIPLoss()
        
        # Initialize LPIPS model
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg', 
            normalize=True
        ).to('cuda')
        
        # Initialize DINO model
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base').to('cuda')
        
        # Initialize FovVideoVDP model
        self.fv = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='threshold')
        
        # Initialize CLIP model
        model_name = 'ViT-L/14@336px'
        image_size = 336
        
        model, _ = clip.load(model_name, device='cuda')
        self.clip = model
        
        # Image transform for CLIP
        self.clip_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                                [0.26862954, 0.26130258, 0.27577711])
        ])

    def add(self, prediction: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        """
        Add a batch of predictions and calculate metrics against ground truth.
        
        Args:
            prediction: Predicted image tensor (B, C, H, W)
            gt: Ground truth image tensor (B, C, H, W)
            
        Returns:
            Dictionary of calculated metrics for this batch
        """
        with torch.no_grad():
            # Validate inputs
            assert torch.isfinite(prediction).all(), "Prediction contains non-finite values"
            
            # Get batch size
            bs = prediction.shape[0]
            self.total_bs += bs
            
            # Calculate metrics
            metrics = self._calculate_metrics(prediction, gt)
            
            # Update accumulated totals
            for key, value in metrics.items():
                if key in self.total:
                    self.total[key] += value * bs
            
            return metrics

    def _calculate_metrics(self, prediction: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all image quality metrics between prediction and ground truth.
        
        Args:
            prediction: Predicted image tensor
            gt: Ground truth image tensor
            
        Returns:
            Dictionary of calculated metrics
        """
        # Calculate HDR metrics
        hdr_flip = self.hdr_flip.forward(prediction, gt).item()
        l1 = L1Loss(prediction, gt).item()
        rel_l1 = relativeL1Loss(prediction, gt).item()
        relative_mse = relativeL2Loss(prediction, gt).item()
        smape = SMAPE(prediction, gt).item()
        tonemapped_relative_mse = tonemappedRelativeMSE(prediction, gt).item()
        
        # Apply tonemapping for LDR metrics
        ldr_prediction = _tonemap(prediction, self.gamma)
        ldr_gt = _tonemap(gt, self.gamma)
        
        # Calculate LDR metrics
        lpips = self.lpips(ldr_prediction, ldr_gt).item()
        ldr_flip = self.ldr_flip.forward(ldr_prediction, ldr_gt).item()
        mean_psnr = psnr(ldr_prediction, ldr_gt, data_range=1.0, reduction='mean').item()
        mean_ssim = ssim(ldr_prediction, ldr_gt, data_range=1.0, reduction='mean').item()
        
        # Calculate FovVideoVDP metric
        fov, _ = self.fv.predict(
            ldr_prediction.squeeze(0), 
            ldr_gt.squeeze(0), 
            dim_order="CHW"
        )
        fov = fov.item()
        
        # Calculate DINOv2 similarity
        dino_sim = self._calculate_dino_similarity(ldr_prediction, ldr_gt)
        
        # Calculate CLIP similarity
        clip_sim = self._calculate_clip_similarity(ldr_prediction, ldr_gt)
        
        # Return all metrics
        return {
            "L1": l1,
            "Relative L1": rel_l1,
            "Relative MSE": relative_mse,
            "Tonemapped Relative MSE": tonemapped_relative_mse,
            "SMAPE": smape,
            "PSNR": mean_psnr,
            "SSIM": mean_ssim,
            "HDR_FLIP": hdr_flip,
            "LDR_FLIP": ldr_flip,
            "LPIPS": lpips,
            "DINO": dino_sim,
            "CLIP": clip_sim,
            "FOV": fov,
        }
    
    def _calculate_dino_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate DINO-based image similarity.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Returns:
            DINO similarity score (0-1)
        """
        # Process the first image
        inputs1 = self.processor(
            images=img2, 
            return_tensors="pt",
            do_rescale=False
        ).to("cuda")
        
        outputs1 = self.dino(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)
        
        # Process the second image
        inputs2 = self.processor(
            images=img1, 
            return_tensors="pt",
            do_rescale=False
        ).to("cuda")
        
        outputs2 = self.dino(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)
        
        # Calculate cosine similarity
        cos = torch.nn.CosineSimilarity(dim=0)
        sim = cos(image_features1[0], image_features2[0]).item()
        
        # Normalize to 0-1 range
        return (sim + 1) / 2
    
    def _calculate_clip_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate CLIP-based image similarity.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Returns:
            CLIP similarity score
        """
        # Apply transforms to images
        c_gt = self.clip_transform(img2.squeeze(0)).unsqueeze(0)
        c_output = self.clip_transform(img1.squeeze(0)).unsqueeze(0)
        
        # Calculate embeddings
        embedding_a = self.clip.encode_image(c_gt)
        embedding_b = self.clip.encode_image(c_output)
        
        # Calculate cosine similarity
        return torch.nn.functional.cosine_similarity(embedding_a, embedding_b).item()

    def getEpochInfo(self) -> Dict[str, float]:
        """
        Get aggregated metrics for the entire epoch.
        
        Returns:
            Dictionary of averaged metrics
        """
        info = {}
        
        # Average standard metrics by total batch size
        if self.total_bs > 0:
            for key in ["L1", "Relative L1", "Relative MSE", "Tonemapped Relative MSE", 
                      "SMAPE", "PSNR", "SSIM", "LPIPS", "DINO", "CLIP", 
                      "LDR_FLIP", "HDR_FLIP", "FOV"]:
                info[key] = self.total[key] / self.total_bs
        
        return info