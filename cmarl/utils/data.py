import os

import torch
from torch import nn
import numpy as np


this_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(f'{this_path}/../../')

def save_model(model: nn.Module, name: str='model'):
    torch.save(model.state_dict(), f'{root_path}/model/{name}.pth')

def load_model(model: nn.Module, name: str='model'):
    model.load_state_dict(torch.load(f'{root_path}/model/{name}.pth'))
    return model

def is_model_found(name: str= 'model') -> bool:
    return os.path.exists(f'{root_path}/model/{name}.pth')

def save_data(data: np.ndarray, name: str='data'):
    np.save(f'{root_path}/data/{name}.npy', data)

def load_data(name: str='data') -> np.ndarray:
    return np.load(f'{root_path}/data/{name}.npy')

def save_tensor(tensor: torch.Tensor, name: str='tensor'):
    torch.save(tensor, f'{root_path}/data/{name}.pt')

def load_tensor(name: str='tensor') -> torch.Tensor:
    return torch.load(f'{root_path}/data/{name}.pt')

def save_dict(data: dict, name: str='dict'):
    torch.save(data, f'{root_path}/data/{name}.pt')

def load_dict(name: str='dict') -> dict:
    return torch.load(f'{root_path}/data/{name}.pt')