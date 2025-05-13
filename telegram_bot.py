import torch
import pandas as pd
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext
from train_model import JokeModel, joke_to_tensor, vocab_size

# Load joke dataset
data = pd.read_csv("jokes.csv")
joke_setups = list(data["Joke"])[:10]  # Pick first 10 jokes as input prompts

# Load trained AI model
model = JokeModel(vocab_size, embed_size=128, hidden_size=256, output_size=100)
model.load_state_dict(torch.load("joke_model.pth"))
model.eval()  # Set model to evaluation mode

# Function to generate AI-powered jokes
async def joke(update: Update, context: CallbackContext):
    input_joke = joke_setups[torch.randint(0, len(joke_setups), (1,)).item()]
    input_tensor = joke_to_tensor(input_joke[:20], vocab_size)  # Use first 20 characters as seed
    output = model(input_tensor.unsqueeze(0))
    predicted_char = chr(output.argmax().item())
    generated_joke = input_joke + predicted_char  # AI-generated joke extension
    await update.message.reply_text(generated_joke)

# Telegram bot setup
def main():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("joke", joke))

    print("AI Joke Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()