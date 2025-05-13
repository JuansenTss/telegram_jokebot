import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# Use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Load dataset
data = pd.read_csv("jokes.csv")
jokes = list(data["Joke"])

# Set vocabulary size
vocab_size = 256

# Convert jokes into tensors using `.encode()` instead of `ord(c)`
def joke_to_tensor(joke, vocab_size):
    tensor = torch.tensor([min(b, vocab_size - 1) for b in joke.encode()], dtype=torch.long)
    return tensor.to(device)

# Create tensor representation of jokes
joke_tensors = [joke_to_tensor(j, vocab_size) for j in jokes]

# Pad all joke tensors to the same length for batch training
padded_joke_tensors = pad_sequence(joke_tensors, batch_first=True, padding_value=0)

# Batch size for faster training
batch_size = 256

# Prepare DataLoader to speed up processing
data_loader = torch.utils.data.DataLoader(padded_joke_tensors, batch_size=batch_size, shuffle=True)

# Define AI joke model
class JokeModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(JokeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embed = self.embedding(x)
        output, _ = self.lstm(embed)
        return self.fc(output[:, -1])

# Initialize model and move to GPU
model = JokeModel(vocab_size, embed_size=128, hidden_size=256, output_size=vocab_size).to(device)

# Define optimizer & loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Enable CuDNN optimization for faster training
torch.backends.cudnn.benchmark = True

# Training loop with batch processing
def train_model(epochs=50):
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            outputs = model(batch)

            # Ensure target values are within range
            targets = torch.tensor([min(b[-1].item(), vocab_size - 1) for b in batch], dtype=torch.long).to(device)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Start training process
train_model()

# Save trained AI model
torch.save(model.state_dict(), "joke_model.pth")
print("Model training complete. Saved as joke_model.pth")