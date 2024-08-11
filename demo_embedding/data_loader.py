import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("demo_embedding/input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text))

        print(f"total tokens: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) / (B * T):.2f} tokens")

        self.current_position = 0

    def next_batch(self):
        buf = self.tokens[
            self.current_position : self.current_position + (self.B * self.T) + 1
        ]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_position += self.B * self.T

        if self.current_position + self.B * self.T > len(self.tokens):
            self.current_position = 0

        return x, y


if __name__ == "__main__":
    data_loader = DataLoaderLite(12, 64)

    for i in range(5):
        print(f"epoch: {i}")
        x, y = data_loader.next_batch()

        print(x[0, :10])
        print(y[0, :10])
