import torch
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext
from train_model import JokeModel, joke_to_tensor, vocab_size  # Import AI model functions

# Load the trained AI model
model = JokeModel(vocab_size, embed_size=128, hidden_size=256, output_size=100)
model.load_state_dict(torch.load("joke_model.pth"))
model.eval()  # Set model to evaluation mode

# Function to generate AI jokes
async def joke(update: Update, context: CallbackContext):
    input_tensor = joke_to_tensor("Why did the chicken ", vocab_size)  # Example setup
    output = model(input_tensor.unsqueeze(0))
    predicted_char = chr(output.argmax().item())
    generated_joke = "Why did the chicken " + predicted_char  # AI-generated joke
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