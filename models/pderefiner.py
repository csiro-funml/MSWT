# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
This code is developed with reference to the following GitHub repo: PDERefiner: https://github.com/pdearena/pdearena/blob/main/scripts/pderefiner_train.py
"""

from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers.schedulers import DDPMScheduler
from .pderefiner_unet import Unet


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, overwrite=False):
        if len(self.shadow) > 0 and not overwrite:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.detach() + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        if len(self.shadow) == 0:
            print("Warning: EMA shadow is empty. Cannot apply shadow.")
        else:
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.data
                    param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

        

class SimpleModel(nn.Module):
    """Simple placeholder model for demonstration"""
    def __init__(self, input_channels, output_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, output_channels, 3, padding=1)
        self.output_channels = output_channels
        self.relu = nn.ReLU()
        
    def forward(self, x, time=None, z=None):
        assert x.dim() == 5
        orig_shape = x.shape

        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x.reshape(
            orig_shape[0], -1, self.output_channels, *orig_shape[3:]
        )


def get_model(args, pde):
    """Simplified model creation function"""
    # Calculate total input/output channels
    total_input = pde.get('n_channels', 0)
    total_output = total_input
    
    
    model = Unet(
        n_channels=total_input,
        time_history=args.get('time_history', 1),
        time_future=args.get('time_future', 1),
        hidden_channels=32,
        activation='gelu',
        norm=True,
    )
    return model


class PDERefiner(nn.Module):
    def __init__(
        self,
        name: str,
        time_history: int,
        time_future: int,
        time_gap: int = 0,
        max_num_steps: int = 1,
        activation: str = "relu",
        padding_mode: str = "zeros",
        predict_difference: bool = False,
        difference_weight: float = 1.0,
        num_refinement_steps: int = 3,
        min_noise_std: float = 4e-7,
        n_spatial_dim: int = 2,
        n_channels: int = 3,
        trajlen: int = 20,
        ema_decay: float = 0.995,
        **kwargs
    ) -> None:
        super().__init__()
        
        # Store hyperparameters
        self.name = name
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap
        self.max_num_steps = max_num_steps
        self.activation = activation
        self.padding_mode = padding_mode
        self.predict_difference = predict_difference
        self.difference_weight = difference_weight
        self.num_refinement_steps = num_refinement_steps
        self.min_noise_std = min_noise_std
        
        # PDE configuration (simplified)
        self.pde = {
            'n_spatial_dim': n_spatial_dim,
            'trajlen': trajlen,
            'n_channels': n_channels
        }
        
        # Set mode based on spatial dimensions
        if n_spatial_dim == 3:
            self._mode = "3D"
            nn.Conv3d = partial(nn.Conv3d, padding_mode=padding_mode)
        elif n_spatial_dim == 2:
            self._mode = "2D"
            nn.Conv2d = partial(nn.Conv2d, padding_mode=padding_mode)
        elif n_spatial_dim == 1:
            self._mode = "1D"
            nn.Conv1d = partial(nn.Conv1d, padding_mode=padding_mode)
        else:
            raise NotImplementedError(f"Spatial dimension {n_spatial_dim} not supported")

        # Create the underlying model
        args = {
            'name': name,
            'time_history': time_history,
            'time_future': time_future,
            'activation': activation,
        }
        self.model = get_model(args, self.pde)
        
        # Loss function
        self.train_criterion = nn.MSELoss()
        
        # EMA for better model stability and performance
        self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
        
        # Diffusion scheduler setup
        betas = [min_noise_std ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        self.time_multiplier = 1000 / num_refinement_steps

        # Validation criterions (simplified)
        self.val_criterions = {"mse": nn.MSELoss()}
        
        # Calculate max start time for rollouts
        time_resolution = trajlen
        reduced_time_resolution = time_resolution - time_history
        self.max_start_time = (
            reduced_time_resolution - time_future * max_num_steps - time_gap
        )
        self.max_start_time = max(0, self.max_start_time)

    def forward(self, x, cond=None):
        """Forward pass through the model"""
        return self.predict_next_solution(x, cond)

    def train_step(self, x, y, cond=None):
        """Single training step for the diffusion model"""
        if self.predict_difference:
            # Predict difference to next step instead of next step directly.
            y = (y - x[:, -1:]) / self.difference_weight
        
        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device)
        noise_factor = self.scheduler.alphas_cumprod.to(x.device)[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(x.ndim - 1)])
        signal_factor = 1 - noise_factor
        noise = torch.randn_like(y)
        y_noised = self.scheduler.add_noise(y, noise, k)
        x_in = torch.cat([x, y_noised], axis=1)
        pred = self.model(x_in, time=k * self.time_multiplier, z=cond)
        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * y
        loss = self.train_criterion(pred, target)
        return loss, pred, target


    def eval_step(self, x, y, cond=None):
        """Single evaluation step"""
        pred = self.predict_next_solution(x, cond)
        loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        return loss, pred, y

    def predict_next_solution(self, x, cond=None):
        """Predict the next solution using the diffusion process"""
        y_noised = torch.randn(
            size=(x.shape[0], self.time_future, *x.shape[2:]), dtype=x.dtype, device=x.device
        )
        for k in self.scheduler.timesteps:
            time = torch.zeros(size=(x.shape[0],), dtype=x.dtype, device=x.device) + k
            x_in = torch.cat([x, y_noised], axis=1)
            pred = self.model(x_in, time=time * self.time_multiplier, z=cond)
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        y = y_noised
        if self.predict_difference:
            y = y * self.difference_weight + x[:, -1:]
        return y
    
    def compute_loss(self, x, y, cond=None):
        """Compute training loss"""
        loss, pred, target = self.train_step(x, y, cond)
        return loss
    
    def get_scalar_vector_losses(self, preds, targets):
        """Compute separate losses for scalar and vector components"""
        n_scalar = self.pde['n_scalar_components']
        n_vector = self.pde['n_vector_components']
        
        scalar_loss = self.train_criterion(
            preds[:, :, 0:n_scalar, ...],
            targets[:, :, 0:n_scalar, ...],
        )
        
        if n_vector > 0:
            vector_loss = self.train_criterion(
                preds[:, :, n_scalar:, ...],
                targets[:, :, n_scalar:, ...],
            )
        else:
            vector_loss = torch.tensor(0.0)
            
        return scalar_loss, vector_loss
    
    def initialize_ema(self):
        """Initialize EMA - call this before training"""
        self.ema.register()
    
    def update_ema(self):
        """Update EMA weights - call this after each training step"""
        self.ema.update()
    
    def apply_ema(self):
        """Apply EMA weights for evaluation/inference"""
        self.ema.apply_shadow()
    
    def restore_original_weights(self):
        """Restore original weights after evaluation"""
        self.ema.restore()
    
    def save_ema_state_dict(self):
        """Get EMA state dict for checkpointing"""
        return self.ema.shadow
    
    def load_ema_state_dict(self, ema_state_dict):
        """Load EMA state dict from checkpoint"""
        self.ema.shadow = ema_state_dict


# Example usage:
if __name__ == "__main__":
    # Example of how to use the PDERefiner class
    model = PDERefiner(
        name="Unetmod-64",
        time_history=4, # T_in
        time_future=1, # T_ar
        time_gap=0,
        max_num_steps=1,  # T_ar, just one step ahead
        n_spatial_dim=2,
        n_channels=3,
        trajlen=14 # T_max
    )
    
    # count the number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Initialize EMA before training
    model.initialize_ema()
    
    # Example input: batch_size=2, time_steps=4, channels=1, height=64, width=64
    x = torch.randn(2, 4, 3, 64, 64)
    y_true = torch.randn(2, 1, 3, 64, 64)  # Ground truth next timestep
    

    # Training step
    model.train()
    loss = model.compute_loss(x, y_true)
    print(f"Training loss: {loss.item()}")
    
    # Update EMA after training step (you would do this in your training loop)
    model.update_ema()
    


    # For evaluation/inference, use EMA weights
    model.eval()
    model.apply_ema()  # Switch to EMA weights
    
    with torch.no_grad():
        prediction = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {prediction.shape}")
    
    model.restore_original_weights()  # Restore original weights
    
    print("✅ EMA integration successful!")
