import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(1337)


# 0. data loading and dataset stats
if Path("input.txt").exists():
    print("File exists")
else:
    resp = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    with open("input.txt", "w") as f: f.write(resp.text)

text = Path("input.txt").read_text()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"dataset sample: {text[:100]}")
print(f"dataset length: {len(text)}")
print(f"vocab_size: {len(set(text))}")
print(f"chars: {''.join(chars)}")

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"{encode('hii there')=}")
print(f"{decode(encode('hii there'))=}")


# 1. torch dataloader
data = torch.tensor(encode(text), dtype=torch.long)
print(f"{data.shape=}, {data.dtype=}")
print(f"{data[:100]=}")

n = int(0.9*len(data))
train_data = data[:n]
valid_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"{t=}: inp={context} target={target}")

## batch construction
batch_size = 4
block_size = 8

def get_batch(data):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch(train_data)
print(f"inputs: {xb.shape=}, {xb=}")
print(f"targets: {yb.shape=}, {yb=}")
print("----------------------")

# 2. training

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            logits = logits.view(batch_size * block_size, vocab_size)
            targets = targets.view(batch_size * block_size)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)

            logits = logits[:, -1, :] # B x C (new char for each batch elem)
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(f"{logits=} {loss=}")

print("untrained output: ", end='')
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optim = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(20000):
    xb, yb = get_batch(train_data)

    logits, loss = m(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()
print("loss: ", loss.item())

# 3. eval
print("trained output: ", end='')
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


