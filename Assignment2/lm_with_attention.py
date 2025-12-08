import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import math
import matplotlib.pyplot as plt


print("loading tokenizer and data...")
tokenizer = Tokenizer.from_file("slovak_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
full_token_ids = torch.load("train_data.pt")
print(f"data loaded. vocab: {vocab_size}, total tokens: {len(full_token_ids)}")

class CustomDataset(Dataset):
    def __init__(self, data_tensor, context_length):
        self.data = data_tensor
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # inputs: from idx to idx + context
        x = self.data[idx : idx + self.context_length]
        
        # targets: shifted by 1 (idx+1 to idx+context+1)
        y = self.data[idx + 1 : idx + self.context_length + 1]
        
        return x, y


def create_dataloader(data, batch_size=128, context_length=32, num_workers=2):
    
    dataset = CustomDataset(data, context_length)
    
    train, dev, test = torch.utils.data.random_split(dataset, [0.8,0.1,0.1])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dev, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


class LanguageModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_heads=4, num_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            batch_first=True,
            dim_feedforward=4*embedding_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        # x shape: [batch, context_length]
        
        # create Embeddings
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        # create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # pass through transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # predict next token
        logits = self.linear(x)
        return logits


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

batch_size = 128
context_length = 32
embedding_dim = 128

train_dataloader, dev_dataloader, test_dataloader = create_dataloader(
    data=full_token_ids, 
    batch_size=batch_size, 
    context_length=context_length, 
)

model = LanguageModelWithAttention(
    vocab_size=vocab_size, 
    embedding_dim=embedding_dim, 
    context_length=context_length,
    num_heads=4,  # 128 / 4 = 32 dim per head
    num_layers=2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

print("starting training...")
train_losses = []
perplexities = []

for epoch in range(num_epochs):
    total_loss = 0
    steps_taken = 0
    model.train()
    
    for batch_idx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        token_logits = logits.view(-1, vocab_size)
        token_labels = y.view(-1)
        loss = criterion(token_logits,token_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps_taken += 1

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        if batch_idx > 3000:
            print("stopping early")
            break


    avg_loss = total_loss / steps_taken
    perplexity = math.exp(avg_loss)
    
    train_losses.append(avg_loss)
    perplexities.append(perplexity)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

    break


model.eval()
total_loss = 0
val_steps = 0

with torch.no_grad():
    for x, y in dev_dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
        val_steps += 1

avg_loss = total_loss / val_steps
perplexity_simple = math.exp(avg_loss)
print(f"perplexity of model with attention: {perplexity_simple:.2f}")


def generate_text(start_text, max_new_tokens=20):
    model.eval()
    ids = tokenizer.encode(start_text).ids
    input_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\ngenerating from: '{start_text}'")
    with torch.no_grad():
        for _ in range(max_new_tokens):
            condensed = input_tensor[:, -context_length:]
            logits = model(condensed)
            next_token = torch.argmax(logits[0, -1, :]).item()
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
    
    return tokenizer.decode(input_tensor[0].tolist())

print(generate_text("Toto je"))