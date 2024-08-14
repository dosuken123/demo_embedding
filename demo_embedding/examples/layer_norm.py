import torch

B, T, C = 2, 3, 4
x = torch.randn(B, T, C)
print(f"x: {x}")
ln = torch.nn.LayerNorm(C)
out = ln(x)
print(f"out: {out}")
