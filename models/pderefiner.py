# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
This code is developed with reference to the following GitHub repo: PDERefiner: https://github.com/pdearena/pdearena/blob/main/scripts/pderefiner_train.py
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
from diffusers.schedulers import DDPMScheduler
from .pderefiner_unet import Unet


def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    loss = F.mse_loss(input, target, reduction="none")
    # avg across space
    reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
    # sum across time + fields
    reduced_loss = reduced_loss.sum(dim=(1, 2))
    # reduce along batch
    if reduction == "mean":
        return torch.mean(reduced_loss)
    elif reduction == "sum":
        return torch.sum(reduced_loss)
    elif reduction == "none":
        return reduced_loss
    else:
        raise NotImplementedError(reduction)
    

class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return custommse_loss(input, target, reduction=self.reduction)


def cond_rollout2d(
    model: torch.nn.Module,
    initial_u: torch.Tensor,
    initial_v: torch.Tensor,
    delta_t: Optional[torch.Tensor],
    cond: Optional[torch.Tensor],
    grid: Optional[torch.Tensor],
    pde: dict,
    time_history: int,
    num_steps: int,
):
    traj_ls = []
    pred = torch.Tensor().to(device=initial_u.device)
    data_vector = torch.Tensor().to(device=initial_u.device)
    for i in range(num_steps):
        if i == 0:
            if pde.n_scalar_components > 0:
                data_scalar = initial_u[:, :time_history]
            if pde.n_vector_components > 0:
                data_vector = initial_v[
                    :,
                    :time_history,
                ]

            data = torch.cat((data_scalar, data_vector), dim=2)

        else:
            data = torch.cat((data, pred), dim=1)
            data = data[
                :,
                -time_history:,
            ]

        if grid is not None:
            data = torch.cat((data, grid), dim=1)

        if delta_t is not None:
            pred = model(data, delta_t, cond)
        else:
            pred = model(data, cond)
        traj_ls.append(pred)

    traj = torch.cat(traj_ls, dim=1)
    return traj



def bootstrap(x: torch.Tensor, Nboot: int, binsize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bootstrapping the mean of tensor.

    Args:
        x (torch.Tensor):
        Nboot (int): _description_
        binsize (int): _description_

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]): bootstrapped mean and bootstrapped variance
    """
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(torch.mean(x[torch.randint(len(x), (len(x),))], axis=(0, 1)))
    return torch.tensor(boots).mean(), torch.tensor(boots).std()


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


def get_model(time_history, time_future, n_channels):
    """Simplified model creation function"""
    
    model = Unet(
        input_channels=n_channels,
        time_history=time_history,
        time_future=time_future,
        # hidden_channels=64,
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
        time_gap: int,
        max_num_steps: int,
        n_spatial_dim: int,
        n_channels: int,
        trajlen: int,
        activation: str,
        criterion: str,
        model: Optional[Dict] = None,
        param_conditioning: Optional[str] = None,
        padding_mode: str = "zeros",
        predict_difference: bool = False,
        difference_weight: float = 1.0,
        num_refinement_steps: int = 3,
        min_noise_std: float = 4e-7,
        ema_decay: float = 0.995,
    ) -> None:
        super().__init__()
        # Set padding for convolutions globally.
        self.n_spatial_dim = n_spatial_dim
        self.predict_difference = predict_difference
        self.difference_weight = difference_weight
        self.time_future = time_future
        self.time_history = time_history
        if (self.n_spatial_dim) == 3:
            self._mode = "3D"
            nn.Conv3d = partial(nn.Conv3d, padding_mode=padding_mode)
        elif (self.n_spatial_dim) == 2:
            self._mode = "2D"
            nn.Conv2d = partial(nn.Conv2d, padding_mode=padding_mode)
        elif (self.n_spatial_dim) == 1:
            self._mode = "1D"
            nn.Conv1d = partial(nn.Conv1d, padding_mode=padding_mode)
        else:
            raise NotImplementedError(f"{self.n_spatial_dim}")

        self.model = get_model(time_history, time_future, n_channels)
        self.train_criterion = CustomMSELoss()
        # For Diffusion models and models in general working on small errors,
        # it is better to evaluate the exponential average of the model weights
        # instead of the current weights. If an appropriate scheduler with
        # cooldown is used, the test results will be not influenced.
        self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
        # We use the Diffusion implementation here. Alternatively, one could
        # implement the denoising manually.
        betas = [min_noise_std ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        self.time_multiplier = 1000 / num_refinement_steps

        self.val_criterions = {"mse": CustomMSELoss()}
        self.rollout_criterions = {"mse": torch.nn.MSELoss(reduction="none")}
        
        time_resolution = trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - time_history
        # Number of future points to predict
        self.max_start_time = (
            reduced_time_resolution - time_future * max_num_steps - time_gap
        )
        self.max_start_time = max(0, self.max_start_time)

    def forward(self, x, cond):
        return self.predict_next_solution(x, cond)

    def train_step(self, batch):
        u_prev, u_t = batch
        if self.predict_difference:
            # Predict difference to next step instead of next step directly.
            u_t = (u_t - u_prev[:, -1:]) / self.difference_weight
        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (u_prev.shape[0],), device=u_prev.device) 
        noise_factor = self.scheduler.alphas_cumprod.to(u_prev.device)[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(u_prev.ndim - 1)])
        signal_factor = 1 - noise_factor
        noise = torch.randn_like(u_t)
        u_t_noised = self.scheduler.add_noise(u_t, noise, k)

        # print("input to the model: ", "u_t shape" ,u_t_noised.shape, "u_prev shape" ,u_prev.shape, "time" ,(k * self.time_multiplier).shape)
        pred = self.model(x=u_prev, z=u_t_noised, time=k * self.time_multiplier)
        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * u_t
        loss = self.train_criterion(pred, target)
        return loss, pred, target

    def eval_step(self, u_prev):
        pred = self.predict_next_solution(u_prev)
        return pred

    def predict_next_solution(self,u_prev):
        y_noised = torch.randn(
            size=(u_prev.shape[0], self.time_future, *u_prev.shape[2:]), dtype=u_prev.dtype, device=u_prev.device
        )
        for k in self.scheduler.timesteps:
            time = torch.zeros(size=(u_prev.shape[0],), dtype=u_prev.dtype, device=u_prev.device) + k
            pred = self.model(x=u_prev, time=time * self.time_multiplier, z=y_noised)
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        y = y_noised
        if self.predict_difference:
            y = y * self.difference_weight + u_prev[:, -1:]
        return y

    def training_step(self, batch):
        loss, preds, targets = self.train_step(batch)
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for key in outputs[0].keys():
            if "loss" in key:
                loss_vec = torch.stack([outputs[i][key] for i in range(len(outputs))])
                mean, std = bootstrap(loss_vec, 64, 1)
                self.log(f"train/{key}_mean", mean)
                self.log(f"train/{key}_std", std)

    def compute_rolloutloss(self, batch: Any):
        (u, v, cond, grid) = batch

        losses = {k: [] for k in self.rollout_criterions.keys()}
        for start in range(
            0,
            self.max_start_time + 1,
            self.hparams.time_future + self.hparams.time_gap,
        ):
            end_time = start + self.hparams.time_history
            target_start_time = end_time + self.hparams.time_gap
            target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps

            init_u = u[:, start:end_time, ...]
            if self.pde.n_vector_components > 0:
                init_v = v[:, start:end_time, ...]
            else:
                init_v = None
            targ_u = u[:, target_start_time:target_end_time, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u

            pred_traj = cond_rollout2d(
                self,
                init_u,
                init_v,
                None,
                cond,
                grid,
                self.pde,
                self.hparams.time_history,
                min(targ_u.shape[1], self.hparams.max_num_steps),
            )
            for k, criterion in self.rollout_criterions.items():
                loss = criterion(pred_traj, targ_traj)
                loss = loss.mean(dim=(0,) + tuple(range(2, loss.ndim)))
                losses[k].append(loss)
        loss_vecs = {k: sum(v) / max(1, len(v)) for k, v in losses.items()}
        return loss_vecs

    def validation_step(self, u_pred: Any, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            # one-step loss
            preds = self.eval_step(u_pred)
            return  preds
            # if self._mode == "1D" or self._mode == "2D":
            #     loss_mse["scalar_mse"] = self.val_criterions["mse"](
            #         preds[:, :, 0 : self.pde.n_scalar_components, ...],
            #         targets[:, :, 0 : self.pde.n_scalar_components, ...],
            #     )
            #     loss_mse["vector_mse"] = self.val_criterions["mse"](
            #         preds[:, :, self.pde.n_scalar_components :, ...],
            #         targets[:, :, self.pde.n_scalar_components :, ...],
            #     )

            #     for k in loss_mse.keys():
            #         self.log(f"valid/loss/{k}", loss_mse[k])
            #     return {f"{k}_loss": v for k, v in loss_mse.items()}

            # else:
            #     raise NotImplementedError(f"{self._mode}")
            

        elif dataloader_idx == 1:
            # rollout loss
            if self._mode == "1D" or self._mode == "2D":
                loss_vecs = self.compute_rolloutloss(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss_mse = loss_vecs["mse"].sum()
            loss_mse_t = loss_vecs["mse"].cumsum(0)
            chan_avg_loss = loss_mse / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss_mse)
            return {
                "unrolled_loss": loss_mse,
                "loss_timesteps": loss_mse_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
                "corr": loss_vecs["corr"],
            }

    def validation_epoch_end(self, outputs: List[Any]):
        if len(outputs) > 1:
            if len(outputs[0]) > 0:
                for key in outputs[0][0].keys():
                    if "loss" in key:
                        loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                        mean, std = bootstrap(loss_vec, 64, 1)
                        self.log(f"valid/{key}_mean", mean)
                        self.log(f"valid/{key}_std", std)

            if len(outputs[1]) > 0:
                unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
                loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
                loss_timesteps = loss_timesteps_B.mean(0)

                log_timesteps = range(0, loss_timesteps.shape[0], max(1, loss_timesteps.shape[0] // 10))

                for i in log_timesteps:
                    self.log(f"valid/intime_{i}_loss", loss_timesteps[i])

                mean, std = bootstrap(unrolled_loss, 64, 1)
                self.log("valid/unrolled_loss_mean", mean)
                self.log("valid/unrolled_loss_std", std)

                # Correlation
                corr_timesteps_B = torch.stack([outputs[1][i]["corr"] for i in range(len(outputs[1]))], dim=0)
                corr_timesteps = corr_timesteps_B.mean(0)
                for threshold in [0.8, 0.9, 0.95]:
                    self.log(f"valid/time_till_corr_lower_{threshold}", (corr_timesteps > threshold).float().sum())
                for t in log_timesteps:
                    self.log(f"valid/corr_at_{t}", corr_timesteps[t])

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "1D" or self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                self.log("test/loss", loss)
                return {f"{k}_loss": v for k, v in loss.items()}
            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            if self._mode == "1D" or self._mode == "2D":
                loss_vecs = self.compute_rolloutloss(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss_mse = loss_vecs["mse"].sum()
            loss_mse_t = loss_vecs["mse"].cumsum(0)
            chan_avg_loss = loss_mse / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss_mse)
            return {
                "unrolled_loss": loss_mse,
                "loss_timesteps": loss_mse_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
                "corr": loss_vecs["corr"],
            }

    def test_epoch_end(self, outputs: List[Any]):
        assert len(outputs) > 1
        if len(outputs[0]) > 0:
            for key in outputs[0][0].keys():
                if "loss" in key:
                    loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                    mean, std = bootstrap(loss_vec, 64, 1)
                    self.log(f"test/{key}_mean", mean)
                    self.log(f"test/{key}_std", std)
        if len(outputs[1]) > 0:
            unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
            loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
            loss_timesteps = loss_timesteps_B.mean(0)
            log_timesteps = range(0, loss_timesteps.shape[0], max(1, loss_timesteps.shape[0] // 10))
            for i in log_timesteps:
                self.log(f"test/intime_{i}_loss", loss_timesteps[i])

            mean, std = bootstrap(unrolled_loss, 64, 1)
            self.log("test/unrolled_loss_mean", mean)
            self.log("test/unrolled_loss_std", std)

            # Correlation
            corr_timesteps_B = torch.stack([outputs[1][i]["corr"] for i in range(len(outputs[1]))], dim=0)
            corr_timesteps = corr_timesteps_B.mean(0)
            for threshold in [0.8, 0.9, 0.95]:
                self.log(f"tests/time_till_corr_lower_{threshold}", (corr_timesteps > threshold).float().sum())
            for t in log_timesteps:
                self.log(f"tests/corr_at_{t}", corr_timesteps[t])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_fit_start(self):
        self.ema.register()

    def on_train_batch_end(self, *args, **kwargs):
        self.ema.update()

    def on_validation_start(self):
        self.apply_ema()

    def on_validation_end(self):
        self.remove_ema()

    def on_test_start(self):
        self.apply_ema()

    def on_test_end(self):
        self.remove_ema()

    def apply_ema(self):
        self.ema.apply_shadow()

    def remove_ema(self):
        self.ema.restore()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.shadow = checkpoint["ema"]



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
        trajlen=14, # T_max
        activation='gelu',
        criterion='mse',
    )
    
    # count the number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Initialize EMA before training
    # model.initialize_ema()
    

    # Example input: batch_size=2, time_steps=4, channels=1, height=64, width=64
    x = torch.randn(2, 4, 3, 64, 64)
    y_true = torch.randn(2, 1, 3, 64, 64)  # Ground truth next timestep
    

    # Training step
    model.train()
    loss = model.training_step((x, y_true))
    # loss = model.compute_loss(x, y_true)
    print(f"Training loss: {loss.item()}")
    loss.backward()


    # For evaluation/inference, use EMA weights
    model.eval()

    with torch.no_grad():
        prediction = model.validation_step(x)
        # compare with the ground truth

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {prediction.shape}")
    
    
    print("✅ EMA integration successful!")
