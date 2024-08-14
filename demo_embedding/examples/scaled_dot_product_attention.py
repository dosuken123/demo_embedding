import torch
import math

# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

B = 1
haead_dim = 2
L, S = 3, 3
E, Ev = 4, 4
query = torch.rand(B, haead_dim, L, E, dtype=torch.float16, device="cpu")
key = torch.rand(B, haead_dim, S, E, dtype=torch.float16, device="cpu")
value = torch.rand(B, haead_dim, S, Ev, dtype=torch.float16, device="cpu")

print(f"query.shape: {query.shape}")
print(f"key.shape: {key.shape}")
print(f"value.shape: {value.shape}")
print("value")
print(value)

# out = torch.nn.functional.scaled_dot_product_attention(query,key,value)
# print(f"out: {out}")
# print(f"out.shape: {out.shape}")

attn_bias = torch.zeros(L, S, dtype=query.dtype)
temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
print("mask")
print(attn_bias)

scale_factor = 1 / math.sqrt(query.size(-1))
attn_weight = query @ key.transpose(-2, -1) * scale_factor
print(f"query @ key")
print(attn_weight)
attn_weight += attn_bias
print("+= attn_bias")
print(attn_weight)
attn_weight = torch.softmax(attn_weight, dim=-1)
attn_weight = torch.dropout(attn_weight, 0.0, train=True)
print("softmax")
print(attn_weight)
ret = attn_weight @ value
print("ret")
print(ret)
print(ret.shape)
