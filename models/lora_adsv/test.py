import loralib as lora
import torch

layer = lora.Linear(2, 3)
layer.eval()
layer.train()

conv = lora.Conv2d(256, 256, 3, groups = 1, r=8)
grouped_conv = lora.Conv2d(256, 256, 3, groups = 2, r=8)
input_tensor = torch.randn((8, 256, 32, 32))

conv(input_tensor)
grouped_conv(input_tensor)