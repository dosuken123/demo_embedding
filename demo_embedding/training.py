from demo_embedding.model import GPT, GPTConfig
from demo_embedding.data_loader import DataLoaderLite
import torch
import matplotlib.pyplot as plt

B = 16
T = 1024
step_num = 50

config = GPTConfig(context_size=T)
model = GPT(config)

data_loader = DataLoaderLite(B, T)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
for i in range(step_num):
    optimizer.zero_grad()

    x, y = data_loader.next_batch()
    logits, loss = model(x, y)

    loss.backward()
    optimizer.step()

    losses.append(loss.detach().numpy())

    print(f"step: {i} | loss: {loss}")

plt.plot(losses)
plt.show()
