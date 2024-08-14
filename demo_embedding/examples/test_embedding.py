import torch


embedding = torch.nn.Embedding(16, 32)

x = torch.tensor([0])
x1_embed = embedding(x)

print(x1_embed)

x = torch.tensor([1])

print(embedding(x))

x = torch.tensor([0])
x3_embed = embedding(x)

print(x3_embed)
print(x1_embed + x3_embed)

assert x1_embed.tolist() == x3_embed.tolist()
assert (x1_embed + x3_embed).tolist() == (x1_embed * 2).tolist()
