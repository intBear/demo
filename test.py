import torch
import torch.nn as nn


def image_to_coordinates(image):
    N, C, H, W = image.shape
    x_coords = torch.linspace(0, 1, W)
    y_coords = torch.linspace(0, 1, H)
    grid = torch.stack(torch.meshgrid(x_coords, y_coords), dim=-1)
    grid = grid.permute(1, 0, 2).unsqueeze(0).repeat(N, 1, 1, 1)
    return grid


a = torch.randn(1, 3, 8, 16)
b = image_to_coordinates(a)
print(b.shape)

