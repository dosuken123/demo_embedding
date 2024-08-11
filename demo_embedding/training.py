from demo_embedding.model import GPT, GPTConfig
from demo_embedding.data_loader import DataLoaderLite
import torch
import time
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    B = 16
    T = 1024
    step_num = 50
    learning_rate = 1e-3
    device = "cuda"
else:
    B = 8
    T = 64
    step_num = 10
    learning_rate = 1e-5
    device = "cpu"

print(f"device: {device}")

config = GPTConfig(context_size=T)
model = GPT(config)
model.to(device)

data_loader = DataLoaderLite(B, T)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def report(func):
    def wrapper(i, *args, **kwargs):
        start_time = time.time()
        loss = func(i, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"step: {i} | loss: {loss:.4f} | elapsed: {elapsed_time*1000:.1f} msec")
        return loss
    return wrapper

@report
def step(i):
    optimizer.zero_grad()
    x, y = data_loader.next_batch()
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    return loss

losses = []
for i in range(step_num):
    loss = step(i)
    if device == 'cuda':
        loss = loss.cpu()
    losses.append(loss.detach().numpy())

plt.plot(losses)
plt.show()
