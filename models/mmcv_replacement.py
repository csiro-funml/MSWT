"""
Replacement for mmcv functions to avoid compatibility issues.
This file provides minimal implementations of the mmcv functions used in the project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List


class ConvModule(nn.Module):
    """
    A conv block that bundles conv/norm/activation layers.
    This is a simplified version of mmcv.cnn.ConvModule.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = None,
        order: tuple = ('conv', 'norm', 'act'),
        **kwargs
    ):
        super(ConvModule, self).__init__()
        
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode
        )
        
        # Normalization layer
        if self.with_norm:
            if norm_cfg.get('type') == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_cfg.get('type') == 'SyncBN':
                self.norm = nn.BatchNorm2d(out_channels)
            else:
                self.norm = nn.BatchNorm2d(out_channels)  # Default to BN
        else:
            self.norm = None
            
        # Activation layer
        if self.with_activation:
            if act_cfg.get('type') == 'ReLU':
                self.activate = nn.ReLU(inplace=True)
            elif act_cfg.get('type') == 'LeakyReLU':
                self.activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            else:
                self.activate = nn.ReLU(inplace=True)  # Default to ReLU
        else:
            self.activate = None
    
    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0):
    """
    Initialize module with normal distribution.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def auto_fp16(apply_to=None, out_fp32=False):
    """
    Decorator to enable automatic mixed precision.
    This is a simplified version that just returns the original function.
    """
    def decorator(func):
        return func
    return decorator


def force_fp32(apply_to=None, out_fp32=False):
    """
    Decorator to force fp32 precision.
    This is a simplified version that just returns the original function.
    """
    def decorator(func):
        return func
    return decorator


class Registry:
    """
    A registry to map strings to classes.
    This is a simplified version of mmcv.utils.Registry.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._module_dict = {}
    
    def register_module(self, name: str = None, module: type = None):
        """Register a module."""
        if module is None:
            # Used as a decorator
            def _register(cls):
                module_name = name if name is not None else cls.__name__
                self._module_dict[module_name] = cls
                return cls
            return _register
        else:
            # Used as a function
            module_name = name if name is not None else module.__name__
            self._module_dict[module_name] = module
            return module
    
    def get(self, name: str):
        """Get a module by name."""
        return self._module_dict.get(name)
    
    def __getitem__(self, name: str):
        return self._module_dict[name]
    
    def __contains__(self, name: str):
        return name in self._module_dict


def build_from_cfg(cfg: Dict, registry: Registry, default_args: Optional[Dict] = None):
    """
    Build a module from config dict.
    This is a simplified version of mmcv.utils.build_from_cfg.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    
    if 'type' not in cfg:
        raise KeyError(f'cfg must contain the key "type", but got {cfg}')
    
    args = cfg.copy()
    obj_type = args.pop('type')
    
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    else:
        obj_cls = obj_type
    
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    
    return obj_cls(**args)
