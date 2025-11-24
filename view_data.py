import torch

data = torch.load("logs/causal_world_reaching/inv/6/pad_finger_link_mass.pt")
print(type(data))
print(data)