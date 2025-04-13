"""Training loss functions.

This module provides various loss functions for comparing predictions with target values
in neural network training contexts.
"""
import torch


EPSILON = 1e-2  


def _tonemap(im):
    """Apply tonemapping to an image tensor.
    
    Args:
        im: Input tensor
        
    Returns:
        Tonemapped tensor
    """
    im = torch.clamp(im, min=0)
    return im / (1+im)


def L1Loss(prediction, target):
    """Calculate L1 (Mean Absolute Error) loss between prediction and target.
    
    Args:
        prediction: The predicted values
        target: The target values
        
    Returns:
        The L1 loss
    """
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return l1_loss(prediction, target)
    

def relativeL1Loss(prediction, target):
    """Calculate relative L1 loss between prediction and target.
    
    This loss normalizes the absolute difference by the magnitude of the target,
    making it scale-invariant.
    
    Args:
        prediction: The predicted values
        target: The target values
        
    Returns:
        The relative L1 loss
    """
    loss = torch.abs(prediction - target) / (
        torch.abs(target.detach()) + EPSILON)

    return torch.mean(loss)
    

def relativeL2Loss(prediction, target):
    """Calculate relative L2 loss between prediction and target.
    
    This loss normalizes the squared difference by the squared magnitude of the target,
    making it scale-invariant.
    
    Args:
        prediction: The predicted values
        target: The target values
        
    Returns:
        The relative L2 loss
    """
    loss = torch.square(prediction - target) / (
        torch.square(target.detach()) + EPSILON)

    return torch.mean(loss)


def SMAPE(prediction, target):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE) between prediction and target.
    
    SMAPE is a scale-independent metric that's symmetric with respect to prediction and target.
    
    Args:
        prediction: The predicted values
        target: The target values
        
    Returns:
        The SMAPE loss
    """
    loss = torch.abs(prediction-target) / (
        torch.abs(prediction.detach()) + torch.abs(target.detach()) + EPSILON)

    return torch.mean(loss)


def tonemappedRelativeMSE(prediction, target):
    """Calculate relative MSE loss between tonemapped prediction and target.
    
    This is useful for HDR image comparison, as tonemapping compresses the dynamic range.
    
    Args:
        prediction: The predicted values
        target: The target values
        
    Returns:
        The tonemapped relative MSE loss
    """
    prediction = _tonemap(prediction)
    target = _tonemap(target)
    loss = torch.square(prediction - target) / (
        torch.square(target.detach() + EPSILON))
    
    return torch.mean(loss)