#!/usr/bin/env python3
"""
Image Denoising Training Script
"""

import os
import json
import time
import logging
import gc
import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import configargparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from typing import Dict, Tuple, List, Optional, Union, Any

# Custom imports
from noisebase.loaders.torch import TrainingSampleLoader_v1 as TrainDataloader
from noisebase.loaders.torch import TestSampleLoader_v1 as TestDataloader
from deepfloyd_if.modules.stage_II import IFStageII
from dn_utils import _tm, _undoTonemap, get_conditioning, BUFFERS, SRC_CONFIG, TEST32_CONFIG, TEST8_CONFIG
from loss import *
from metric_logger import MetricLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clean_memory() -> None:
    """Aggressively clean GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Config:
    """Configuration management for the training script."""
    
    def __init__(self) -> None:
        """Initialize and parse command line arguments."""
        self.parser = configargparse.ArgParser()
        self._add_arguments()
        self.args = self.parser.parse_args()
        
        # Derived settings
        self.device = 'cuda:0'
        self.date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        
        # Setup prefix based on mode
        if self.args.eval:
            self.prefix = f'eval-CN-{self.args.doCN}_tm-{self.args.tm}_spp-{self.args.spp}_{self.args.steps}_tstart-{self.args.support_noise_less_qsample_steps}_'
        else:
            self.prefix = f'train-{self.args.model}_LR-{self.args.lr}_sched-{self.args.lr_scheduler}_time_'
        
        # Log base path
        self.log_path_base = os.path.join(
            self.args.logdir, 
            self.prefix + self.date_time
        )
        
        # Epsilon for optimizer
        self.eps = 1e-7 if self.args.dtype in ['bfloat16', 'fp16'] else 1e-8

    def _add_arguments(self) -> None:
        """Add all command line arguments."""
        p = self.parser
        
        # Training parameters
        p.add('--steps', default='super27', type=str, 
              help='Number of steps for stage inference')
        p.add('--sample_loop', default='ddpm', type=str, 
              help='ddpm or ddim')
        p.add('--tm', default=1, type=int, 
              help='tonemapper during inference: log10 (0); ln (1) and gamma (2).')
        p.add('--logdir', required=True, 
              help='Where to save results')
        p.add('--spp', required=False, default=4, help='spp', type=int)
        p.add('--lr', default=0.00002, type=float, 
              help='learning rate')
        p.add('--batch_size', default=10, type=int, 
              help='training batch size')
        p.add('--num_epochs', default=50, type=int, 
              help='number of epochs')
        p.add('--compile', action='store_true', default=False,
              help='Use torch.compile for model optimization. Does not work as-is with mem efficient attn.')
        p.add('--compile_mode', default='default', type=str, choices=["default", "reduce-overhead", "max-autotune"],
              help='Compilation mode for torch.compile')
        p.add('--gradient_accumulation_steps', default=1, type=int,
              help='Number of steps to accumulate gradients')
        p.add('--lr_scheduler', default='cosine_warmup', type=str, 
              choices=["cosine_warmup", "cosine_restart", "linear", "step"],
              help='Learning rate scheduler type')
        p.add('--warmup_epochs', default=1, type=int,
              help='Number of epochs for warmup')
        p.add('--early_stopping', action='store_true', default=False,
              help='Use early stopping based on validation metrics')
        p.add('--early_stopping_patience', default=3, type=int,
              help='Patience for early stopping')
        
        # Model configuration
        p.add('--model', default='small', choices=('big', 'small'), 
              help='big or small model')
        p.add('--model_load_path', default=None,
              help='pretrained model, None means train from scratch.')
        p.add('--aux_channels', default=11, type=int,
              help='number of aux channels.')
        p.add('--training_res', default=256, type=int,
              help='Resolution of training images.')
        p.add('--dtype', default='fp32', 
              choices=("fp16", "fp32", "bfloat16", "fp64"), 
              help='dtype to train - fp16, fp32, bfloat16, fp64')
        p.add('--gradient_checkpointing', action='store_true', default=False,
              help='Use gradient checkpointing to save memory during training')
        
        # Regularization parameters
        p.add('--aug_level', default=0.0, type=float, 
              help='aug level for stage 2')
        p.add('--dynamic_thresholding_p', default=0.97, type=float, 
              help='dynamic thresholding p')
        p.add('--dynamic_thresholding_c', default=1.5, type=float, 
              help='dynamic thresholding c')
        p.add('--weight_decay', default=0.01, type=float,
              help='Weight decay for AdamW optimizer')
        
        # Training options
        p.add('--amp', action='store_true', 
              help='use automatic mixed precision')
        p.add('--partialTrainable', action='store_true', default=False,
              help='only beginning CN layers trainable')
        p.add('--bias_tsteps', action='store_true', default=False,
              help='Bias training tsteps towards super27 timesteps')
        
        # Inference/evaluation options
        p.add('--eval', action='store_true',
              help='Run in evaluation mode - do not train')
        p.add('--ext', default='.png',
              help='Format to save results e.g. .png or .jpg')
        p.add('--process_single_frame', action='store_true',
              help='Evaluate on just the first frame of the sequence for debugging')
        p.add('--support_noise_less_qsample_steps', default=12, type=int,
              help='At inference time, add noise to low spp render')
        p.add('--profile', action='store_true', default=False,
              help='Profile memory usage during training/inference')
        
        # Model architecture options
        p.add('--doCN', action='store_true', default=False,
              help='Whether to instantiate a CN module')
        
        # Data options
        p.add('--data_path', default=None, 
              help='path to noisebase dataset')
        p.add('--num_workers', default=None, type=int,
              help='Number of dataloader workers, defaults to min(6, cpu_count)')

    def save_config(self) -> None:
        """Save the configuration to a JSON file."""
        if not os.path.exists(self.log_path_base):
            Path(self.log_path_base).mkdir(parents=True, exist_ok=True)
            
        config_path = os.path.join(self.log_path_base, 'commandline_args.txt')
        with open(config_path, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def make_output_dirs(self) -> Tuple[str, str]:
        """
        Create output directories for validation and test images.
        
        Returns:
            Tuple containing paths to test and validation image directories
        """
        path_test = os.path.join(self.log_path_base, 'test_imgs')
        path_val = os.path.join(self.log_path_base, 'val_imgs')
        
        for path in [path_test, path_val]:
            if not os.path.exists(path):
                os.makedirs(path)
                
        return path_test, path_val


class ModelManager:
    """Manages model initialization, configuration and training."""
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the model manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.args = config.args
        self.device = config.device
        self.text_prompts = self._load_text_prompts()
        self.model = None
        
    def _load_text_prompts(self) -> torch.Tensor:
        """
        Load empty text prompts for model input.
        
        Returns:
            Tensor of text prompts
        """
        prompts = torch.from_numpy(
            np.load('empty_prompt_1_77_4096.npz', allow_pickle=True)['arr']
        ).to(self.device)
        
        return prompts.repeat(self.args.batch_size, 1, 1)
    
    def initialize_model(self) -> IFStageII:
        """
        Initialize and configure the model.
        
        Returns:
            Initialized model instance
        """
        logger.info(f"Initializing Stage 2 {'small' if self.args.model == 'small' else 'large'} model")
        
        model_path = 'IF-II-M-v1.0' if self.args.model == 'small' else 'IF-II-L-v1.0'
        
        aux_channels = self.args.aux_channels
            
        model_kwargs = {
            'doCN': self.args.doCN,
            'aux_ch': aux_channels,
            'attention_resolutions': '32,16'
        }
        
        self.model = IFStageII(
            model_path,
            device=self.device, 
            filename=self.args.model_load_path, 
            model_kwargs=model_kwargs
        )
        
        self._set_model_dtype()
        
        # Enable gradient checkpointing if requested
        if self.args.gradient_checkpointing and hasattr(self.model.model, 'enable_gradient_checkpointing'):
            logger.info("Enabling gradient checkpointing")
            self.model.model.enable_gradient_checkpointing()
                
        # Apply torch.compile if requested
        if self.args.compile:
            if hasattr(torch, 'compile'):
                logger.info(f"Compiling model with mode: {self.args.compile_mode}")
                try:
                    self._compile_model()
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}. Continuing with uncompiled model.")
            else:
                logger.warning("torch.compile not available in this PyTorch version. Please upgrade to PyTorch 2.0+.")
                
        logger.info(f"Model dtype: {self.model.model.dtype}")
        return self.model
    
    def _compile_model(self) -> None:
        """Apply torch.compile to the model for faster training and inference."""
        # Different compile options for specific use cases
        compile_options = {
            "default": {},
            "reduce-overhead": {"mode": "reduce-overhead"},
            "max-autotune": {"mode": "max-autotune"}
        }
        
        try:
            # Main model 
            self.model.model = torch.compile(
                self.model.model, 
                **compile_options[self.args.compile_mode]
            )
            
            # Compile control model if it exists
            if self.args.doCN and hasattr(self.model.model, 'control_model'):
                self.model.model.control_model = torch.compile(
                    self.model.model.control_model,
                    **compile_options[self.args.compile_mode]
                )
                
            logger.info("Model successfully compiled")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
            raise
    
    def _set_model_dtype(self) -> None:
        """Set the model's data type based on configuration."""
        dtype_mapping = {
            'bfloat16': (torch.bfloat16, 'bf16'),
            'fp16': (torch.float16, '16'),
            'fp32': (torch.float32, '32'),
            'fp64': (torch.float64, '64')
        }
        
        if self.args.dtype in dtype_mapping:
            dtype, precision = dtype_mapping[self.args.dtype]
            self.model.model.dtype = dtype
            self.model.model.precision = precision
            
            # Handle control model if it exists
            if self.args.doCN and hasattr(self.model.model, 'control_model'):
                self.model.model.control_model.dtype = dtype
                self.model.model.control_model.precision = precision
            
            # Convert all parameters to the specified dtype
            for _, p in self.model.model.named_parameters():
                p.data = p.type(dtype)
    
    def prepare_trainable_parameters(self) -> List[torch.Tensor]:
        """
        Prepare trainable parameters for optimization.
        
        Returns:
            List of parameters to optimize
        """
        # First set all parameters as non-trainable
        for _, p in self.model.model.named_parameters():
            p.requires_grad = False
        
        params = []
        
        # Configure control model parameters for training if applicable
        if self.args.doCN:
            for name, param in self.model.model.control_model.named_parameters():
                # Skip certain layers
                if 'encoder_pooling' in name or 'encoder_proj' in name:
                    continue
                
                # If partial training, only include specific layers
                if self.args.partialTrainable:
                    if (name != 'input_blocks.0.0.weight' and 
                        not name.startswith('zero_convs.') and 
                        not name.startswith('input_hint_block.') and 
                        not name.startswith('middle_block_out.')):
                        continue
                
                param.requires_grad = True
                params.append(param)
                
        return params
    
    def create_optimizer(self, parameters: List[torch.Tensor]) -> torch.optim.Optimizer:
        """
        Create optimizer for model training.
        
        Args:
            parameters: Parameters to optimize
            
        Returns:
            Configured optimizer
        """
        return torch.optim.AdamW(
            parameters, 
            eps=self.config.eps, 
            lr=float(self.args.lr),
            weight_decay=self.args.weight_decay
        )
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, dataloader_length: int) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            dataloader_length: Length of the dataloader for step calculation
            
        Returns:
            Configured learning rate scheduler
        """
        # Account for gradient accumulation in step calculation
        actual_dataloader_length = dataloader_length // self.args.gradient_accumulation_steps
        total_steps = actual_dataloader_length * self.args.num_epochs
        warmup_steps = actual_dataloader_length * self.args.warmup_epochs
        
        if self.args.lr_scheduler == "cosine_warmup":
            # Cosine annealing with warmup - good for diffusion models
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_steps - warmup_steps,
                eta_min=1e-5 * self.args.lr
            )
            
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler], 
                milestones=[warmup_steps]
            )
            
        elif self.args.lr_scheduler == "cosine_restart":
            # Cosine with restarts - can help escape local minima
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=dataloader_length * 2,  # Restart every 2 epochs
                T_mult=2,  # Double the restart interval after each restart
                eta_min=1e-5 * self.args.lr
            )
            
        elif self.args.lr_scheduler == "linear":
            # Simple linear decay
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: 1.0 - step / total_steps
            )
            
        else:  # "step" scheduler
            # Step decay (similar to original script)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=dataloader_length * 4,  # Every 4 epochs
                gamma=0.5
            )
        
        return scheduler
    
    def get_diffusion(self) -> Any:
        """
        Get diffusion model from stage 2.
        
        Returns:
            Diffusion model instance
        """
        return self.model.get_diffusion(None)


class Trainer:
    """Handles model training and inference."""
    
    def __init__(self, config: Config, model_manager: ModelManager) -> None:
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
            model_manager: ModelManager instance
        """
        self.config = config
        self.args = config.args
        self.model_manager = model_manager
        self.model = model_manager.model
        self.path_test, self.path_val = config.make_output_dirs()
        self.dataloaders = self._initialize_dataloaders()
        
        # More aggressive mixed precision with automatic scaling tuning
        self.scaler = GradScaler(
            enabled=self.args.amp,
            init_scale=2**10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100
        )
        
    def _initialize_dataloaders(self) -> Dict[str, Union[TrainDataloader, TestDataloader]]:
        """
        Initialize training and validation dataloaders.
        
        Returns:
            Dictionary of dataloaders
        """
        seed = 42  # Fixed seed for reproducibility
        val_split = 28./1024
        
        # Set optimal number of workers
        if self.args.num_workers is None:
            num_workers = min(6, os.cpu_count() or 6)
        else:
            num_workers = self.args.num_workers
            
        logger.info(f'Using {num_workers} workers for data loading')
        dataloaders = {}

        # Training dataloader
        dataloaders['train'] = TrainDataloader(
            src=SRC_CONFIG, 
            stage='train',
            shuffle=True,
            batch_size=self.args.batch_size, 
            drop_last=True,
            flip_rotate=True, 
            seed=seed, 
            val_split=val_split, 
            num_workers=num_workers,
            samples=32, 
            buffers=BUFFERS, 
            data_path=self.args.data_path
        )

        # Validation dataloader
        dataloaders['val'] = TrainDataloader(
            src=SRC_CONFIG, 
            stage='val',
            shuffle=True,
            batch_size=1, 
            drop_last=False,
            flip_rotate=False, 
            seed=seed, 
            val_split=val_split, 
            num_workers=2,  # Fewer workers for validation
            samples=32, 
            buffers=BUFFERS, 
            data_path=self.args.data_path
        )
        
        # Test dataloader (only if in eval mode)
        if self.args.eval:
            cfg = TEST32_CONFIG if self.args.spp > 8 else TEST8_CONFIG
            cfg["samples"] = self.args.spp
            
            dataloaders['test'] = TestDataloader(
                src=cfg, 
                samples=self.args.spp, 
                buffers=BUFFERS, 
                data_path=self.args.data_path, 
                save_dir="tmp/unused", 
                output="unused/{sequence_name}/frame{index:04d}.png", 
                process_single_frame=self.args.process_single_frame
            )
            
        return dataloaders
    
    @torch.no_grad()
    def run_inference(self, epoch: int, samples: int, loader_type: str = 'val') -> Dict[str, float]:
        """
        Run inference on validation or test dataset.
        
        Args:
            epoch: Current epoch number
            samples: Number of samples per pixel
            loader_type: Dataloader type ('val' or 'test')
            
        Returns:
            Evaluation metrics
        """
        # Set model to evaluation mode
        self.model.model.eval()
        
        # Profile memory if requested
        if self.args.profile:
            start_mem = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"Starting inference with {start_mem:.2f}MB allocated")
        
        is_test = loader_type == 'test'
        loader = self.dataloaders[loader_type]
        
        metric_logger = MetricLogger(gamma=1.0)
        output_path = self.path_test if is_test else self.path_val
        current_path = os.path.join(output_path, str(epoch), f'spp_{samples}')
        Path(current_path).mkdir(parents=True, exist_ok=True)
        
        skip_iters_time = 1
        
        if is_test:
            iteration = 0
            for sequence in loader:
                for _, sample in enumerate(tqdm(sequence.frames)):
                    self._process_sample(
                        iteration=iteration, 
                        sample=sample, 
                        samples=samples, 
                        metric_logger=metric_logger, 
                        output_path=current_path, 
                        skip_iters_time=skip_iters_time, 
                        seq_name=sequence.sequence['name']
                    )
                    iteration += 1
                    
                    # Clean memory periodically during test
                    if iteration % 10 == 0:
                        clean_memory()
        else:
            for iteration, sample in enumerate(tqdm(loader)):
                self._process_sample(
                    iteration=iteration, 
                    sample=sample, 
                    samples=samples, 
                    metric_logger=metric_logger, 
                    output_path=current_path, 
                    skip_iters_time=skip_iters_time
                )
                
                # Clean memory periodically during validation
                if iteration % 10 == 0:
                    clean_memory()
        
        # Final clean up
        clean_memory()
        
        # Profile memory if requested  
        if self.args.profile:
            end_mem = torch.cuda.memory_allocated() / 1024**2
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(f"Inference complete: Current: {end_mem:.2f}MB, Peak: {peak_mem:.2f}MB")
        
        # Log metrics
        epoch_metrics = metric_logger.getEpochInfo()
        log_type = 'test' if is_test else 'val'
        logger.info(f"Epoch {epoch} {log_type} metrics: {epoch_metrics}")
        
        # Save metrics to file
        metrics_file = os.path.join(output_path, f'metrics_{log_type}.txt')
        with open(metrics_file, 'a') as f:
            timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            f.write(f'\nepoch: {epoch}, spp: {samples}, time: {timestamp}\n')
            f.write(f'{epoch_metrics}\n')
            
        return epoch_metrics
    
    @torch.no_grad()
    def _process_sample(
        self, 
        iteration: int, 
        sample: Dict[str, Any], 
        samples: int, 
        metric_logger: MetricLogger, 
        output_path: str, 
        skip_iters_time: int, 
        seq_name: Optional[str] = None
    ) -> None:
        """
        Process a single sample during inference.
        
        Args:
            iteration: Current iteration number
            sample: Sample to process
            samples: Number of samples per pixel
            metric_logger: Metrics logger instance
            output_path: Path to save output images
            skip_iters_time: Number of iterations to skip for timing
            seq_name: Sequence name for test samples
        """
        # Get frame number
        frame_num = sample['frame_index']
        if isinstance(frame_num, list):
            frame_num = frame_num[0][0][0].numpy()
        else:
            frame_num = frame_num.cpu().item()
            
        # Skip non-first frames during validation
        if seq_name is None and frame_num > 0:
            return
            
        if iteration % 10 == 0:
            logger.info(f"Processing iteration {iteration}")
            
        # Process sample with mixed precision if enabled
        with autocast(dtype=torch.float16, enabled=self.args.amp):
            # Prepare input data
            reference = sample['reference'].clone().cuda().clamp(0, 8)
            color, conditioning, _ = get_conditioning(
                sample, 
                min_samples=samples, 
                max_samples=samples, 
                tonemapper=self.args.tm
            )
            
            # Time the inference if not in first few iterations
            if iteration > skip_iters_time:
                start = time.time()
                
            # Run inference
            sample_out, metadata = self.model.embeddings_to_image(
                sample_timestep_respacing=str(self.args.steps), 
                low_res=2*color-1, 
                support_noise=2*color-1,
                support_noise_less_qsample_steps=self.args.support_noise_less_qsample_steps, 
                seed=None, 
                t5_embs=self.model_manager.text_prompts[0:1, ...], 
                hint=2*conditioning-1, 
                aug_level=self.args.aug_level, 
                sample_loop=self.args.sample_loop, 
                dynamic_thresholding_p=self.args.dynamic_thresholding_p,
                dynamic_thresholding_c=self.args.dynamic_thresholding_c
            )
            
            if iteration > skip_iters_time:
                end_time = time.time() - start
                logger.debug(f"Inference time: {end_time:.4f}s")
            
            # Process output
            sample_out = (sample_out + 1) / 2
            sample_out = _undoTonemap(sample_out, self.args.tm)
            sample_out = torch.nan_to_num(sample_out).clamp(0, 8)
            
            # Calculate metrics
            metrics = metric_logger.add(sample_out, reference)
        
        # Generate visualization
        diff = torch.abs(sample_out - reference)
        diff = torch.sum(diff, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        color = _tm(_undoTonemap(color, self.args.tm))
        
        # Create directory structure for output
        save_dir = os.path.join(output_path, '' if seq_name is None else seq_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual sample
        saved_sample = (_tm(sample_out).cpu().numpy().squeeze().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        saved_sample = Image.fromarray(saved_sample)
        saved_sample.save(os.path.join(save_dir, f'IF_{frame_num}{self.args.ext}'))
        
        # Create comparison visualization
        color, diff, sample_out_tm, ref_tm = map(
            lambda y: F.pad(y, (5, 5, 5, 5)), 
            [color, diff, _tm(sample_out), _tm(reference)]
        )
        
        save_out = torch.cat([color, diff, sample_out_tm, ref_tm], dim=-1) * 2 - 1
        
        # Save comparison image
        save_path = os.path.join(save_dir, f'{iteration}{self.args.ext}')
        self.model.to_images(save_out, disable_watermark=True)[0].save(save_path)
        
        # Clean up to free memory
        del sample_out, metadata, diff, color, sample_out_tm, ref_tm, save_out, saved_sample
    
    def train(self, optimizer: torch.optim.Optimizer, params: List[torch.Tensor]) -> None:
        """
        Train the model.
        
        Args:
            optimizer: Optimizer instance
            params: Parameters to optimize
        """
        if self.args.eval:
            logger.info("Running in evaluation mode only")
            self.run_inference(0, self.args.spp, 'test')
            return
        
        diffusion = self.model_manager.get_diffusion()
        best_psnr = 0
        best_epoch = 0
        val_every = 1
        print_freq = 100
        patience_counter = 0
        
        # Create and initialize learning rate scheduler
        scheduler = self.model_manager.create_scheduler(
            optimizer, 
            len(self.dataloaders['train'])
        )
        
        logger.info(f"Starting training with {self.args.num_epochs} epochs")
        logger.info(f"Using {self.args.lr_scheduler} learning rate scheduler")

        # Validate on different SPP values
        for spp in [2, 32]:
            val_metrics = self.run_inference(-1, spp, 'val')
        
        current_psnr = val_metrics['PSNR']

        for epoch in range(self.args.num_epochs):
            logger.info(f"Starting epoch {epoch}")
            self.dataloaders['train'].batch_sampler.epoch = epoch
            self.dataloaders['train'].batch_sampler.shuffle_indices()
            self.model.model.train()
            
            # Update dataloader epoch seed for better randomization
            self.dataloaders['train'].batch_sampler.epoch_idx = int(time.time()) + epoch
            
            total_loss = 0
            step = 0
            
            # Profile memory if requested
            if self.args.profile:
                start_mem = torch.cuda.memory_allocated() / 1024**2
                logger.info(f"Starting epoch with {start_mem:.2f}MB allocated")
            
            for iteration, sample in enumerate(tqdm(self.dataloaders['train'])):
                with autocast(dtype=torch.float16, enabled=self.args.amp):
                    # Get training data
                    color, conditioning, reference = get_conditioning(
                        sample, 
                        min_samples=1, 
                        max_samples=32, 
                        tonemapper=self.args.tm
                    )
                    
                    # Calculate loss
                    loss = diffusion.training_losses(
                        self.model.model, 
                        x_start=2*reference-1,  
                        model_kwargs={
                            'text_emb': self.model_manager.text_prompts, 
                            'low_res': 2*color-1, 
                            'hint': 2*conditioning-1, 
                            'aug_level': self.args.aug_level
                        }, 
                        bias_tsteps=self.args.bias_tsteps
                    )
                    loss_value = loss['loss'].mean() / self.args.gradient_accumulation_steps
                
                # Backpropagation with gradient scaling
                self.scaler.scale(loss_value).backward()
                
                # Log progress
                total_loss += loss_value.item() * self.args.gradient_accumulation_steps
                
                # Update weights after accumulation steps
                if (iteration + 1) % self.args.gradient_accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    step += 1
                
                # Log average loss periodically
                if iteration % print_freq == print_freq - 1:
                    avg_loss = total_loss / print_freq
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(f'Epoch {epoch}, iteration {iteration}, avg loss: {avg_loss:.6f}, lr: {lr:.8f}')
                    total_loss = 0
                
                # Clean memory periodically
                if iteration % 100 == 99:
                    clean_memory()

                # Validate on different SPP values
                if  iteration % 1000 == 999:
                    for spp in [2, 32]:
                        val_metrics = self.run_inference(epoch, spp, 'val')
            
            # Run validation if needed
            if epoch % val_every == 0:
                # Validate on different SPP values
                for spp in [2, 32]:
                    val_metrics = self.run_inference(epoch, spp, 'val')
                
                # Save best model
                current_psnr = val_metrics['PSNR']
                
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    best_epoch = epoch
                    patience_counter = 0
                    best_model_path = os.path.join(self.config.log_path_base, 'best.pt')
                    logger.info(f"Saving best model with PSNR {best_psnr:.4f} to {best_model_path}")
                    if hasattr(self.model.model, '_orig_mod'):
                        logger.info("Saving state dict from the original uncompiled model")
                        model_state_dict = self.model.model._orig_mod.state_dict()
                    else:
                        model_state_dict = self.model.model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state_dict,
                        'psnr': best_psnr,
                    }, best_model_path)
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} validations. Best PSNR: {best_psnr:.4f} from epoch {best_epoch}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.config.log_path_base, 'ckpt.pt')
            if hasattr(self.model.model, '_orig_mod'):
                logger.info("Saving state dict from the original uncompiled model")
                model_state_dict = self.model.model._orig_mod.state_dict()
            else:
                model_state_dict = self.model.model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'psnr': best_psnr if 'PSNR' in val_metrics else 0,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")
            
            # Early stopping check
            if self.args.early_stopping and patience_counter >= self.args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Profile memory if requested
            if self.args.profile:
                end_mem = torch.cuda.memory_allocated() / 1024**2
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                logger.info(f"Epoch complete: Current: {end_mem:.2f}MB, Peak: {peak_mem:.2f}MB")
                
            # Clean memory at end of epoch
            clean_memory()


def main() -> None:
    """Main function to run the training/evaluation script."""
    try:
        # Initialize configuration
        config = Config()
        config.save_config()
        
        # Initialize model manager and model
        model_manager = ModelManager(config)
        model = model_manager.initialize_model()
        
        # Get trainable parameters
        trainable_params = [] if config.args.eval else model_manager.prepare_trainable_parameters()
        
        # Create optimizer
        optimizer = None if config.args.eval else model_manager.create_optimizer(trainable_params)
        
        # Initialize trainer
        trainer = Trainer(config, model_manager)
        
        # Run training or evaluation
        trainer.train(optimizer, trainable_params)
        
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()