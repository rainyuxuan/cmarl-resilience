import os

import torch
from torch import nn
import numpy as np

def save_model(model: nn.Module, name: str='model'):
    torch.save(model.state_dict(), f'model/{name}.pth')

def load_model(model: nn.Module, name: str='model'):
    model.load_state_dict(torch.load(f'model/{name}.pth'))
    return model

def is_model_found(name: str= 'model') -> bool:
    return os.path.exists(f'model/{name}.pth')

def save_data(data: np.ndarray, name: str='data'):
    np.save(f'data/{name}.npy', data)

def load_data(name: str='data') -> np.ndarray:
    return np.load(f'data/{name}.npy')

def save_tensor(tensor: torch.Tensor, name: str='tensor'):
    torch.save(tensor, f'data/{name}.pt')

def load_tensor(name: str='tensor') -> torch.Tensor:
    return torch.load(f'data/{name}.pt')

def save_dict(data: dict, name: str='dict'):
    torch.save(data, f'data/{name}.pt')

def load_dict(name: str='dict') -> dict:
    return torch.load(f'data/{name}.pt')