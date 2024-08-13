import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
print(f"vocab_size: {tokenizer.vocab_size}")

model = torch.nn.Sequential(
    torch.nn.Embedding(tokenizer.vocab_size, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, tokenizer.vocab_size),
)

input = "This is a apple"
token_ids = tokenizer.encode(input)
x = torch.tensor(token_ids[:-1])[None, :]
print(f"x: {x}")
y = torch.tensor(token_ids[1:])[None, :]
print(f"y: {y}")

logits = model(x)
print(f"logits: {logits.shape}")

loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

print(f"token_ids: {token_ids}")
print(f"logits: {logits}")
print(f"loss: {loss}")

print(f"logits[:, -1, :]: {logits[:, -1, :]}")
prob = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
pred_token_id = torch.multinomial(prob, num_samples=1)
print(f"pred_token_id: {pred_token_id}")
print(pred_token_id.detach().tolist())
pred_output = tokenizer.decode(pred_token_id.detach().tolist()[0])
print(f"{input}{pred_output}")

optimizer = torch.optim.AdamW(model.parameters())
for step in range(1000):
    optimizer.zero_grad()

    if step % 2 == 0:
        banana_input = "This is a banana"
        banana_token_ids = tokenizer.encode(banana_input)
        banana_x = torch.tensor(banana_token_ids[:-1])[None, :]
        logits = model(banana_x)
        banana_y = torch.tensor(banana_token_ids[1:])[None, :]
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), banana_y.view(-1))
    else:
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
    
    loss.backward()
    optimizer.step()

    prob = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
    pred_token_id = torch.multinomial(prob, num_samples=1)
    pred_output = tokenizer.decode(token_ids[:-1] + pred_token_id.detach().tolist()[0])
    print(f"step: {step} | loss: {loss:.2f} | {pred_output}")


