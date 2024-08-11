from demo_embedding.model import GPT, GPTConfig




config = GPTConfig()
model = GPT(config)

# x = torch.randint(0, 50257, (4, 12), dtype=torch.long)
# y = torch.randint(0, 50257, (4, 12), dtype=torch.long)
# print(x)

logits, loss = model(x, y)
print(f"logits: {logits}")
print(f"loss: {loss}")
