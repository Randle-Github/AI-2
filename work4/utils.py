import torch
import torch.nn.functional as F
import sys

def sin_time_embeding(t, number_channels = 256, device = "cuda"):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, number_channels, 2).float() / number_channels)).to(device)
    pos_enc_a = torch.sin(t.repeat(1, number_channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, number_channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

class beta_schedule():
  def __init__(self, beta_start, beta_end, number_timesteps):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.number_timesteps = number_timesteps

  def linear(self):
    return torch.linspace(self.beta_start, self.beta_end, self.number_timesteps)

  def quadratic(self):
    return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.number_timesteps) ** 2

  def sigmoid(self):
    betas = torch.linspace(-6, 6, self.number_timesteps)
    return (torch.sigmoid(betas) * (self.beta_end - self.beta_start)) + self.beta_start

  def cosine(self, s=0.008):
    steps = self.number_timesteps + 1
    x = torch.linspace(0, self.number_timesteps, steps)
    alphas_cumprod = torch.cos(((x / self.number_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, self.beta_start, self.beta_end)
