import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load dataset
data = pd.read_csv("jokes.csv")

# Prepare text data
jokes = list(data["punchline"])
vocab_size = 1000  # Adjust as needed

# Convert jokes into tensors
def joke_to_tensor(joke, vocab_size):
    return torch.tensor([ord(c) % vocab_size for c in joke])

joke_tensors = [joke_to_tensor(j, vocab_size) for j in jokes]

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

# Initialize model
model = JokeModel(vocab_size, embed_size=128, hidden_size=256, output_size=100)

# Define optimizer & loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
def train_model(epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for joke_tensor in joke_tensors:
            optimizer.zero_grad()
            outputs = model(joke_tensor.unsqueeze(0))
            loss = criterion(outputs, joke_tensor[-1].unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Train for 10 epochs
train_model(epochs=10)

# Save the trained model
torch.save(model.state_dict(), "joke_model.pth")
print("Model training complete. Saved as joke_model.pth")