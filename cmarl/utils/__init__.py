from .data import *
from .team import *
from .viz import *


def compute_output_dim(input_dim, kernel_size, stride, padding):
    return (input_dim - kernel_size + 2 * padding) // stride + 1