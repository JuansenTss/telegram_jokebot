import random
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext

# List of jokes
jokes = [
    "Why don’t skeletons fight each other? Because they don’t have the guts!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "What do you call fake spaghetti? An impasta!"
]

# Function to send jokes
async def joke(update: Update, context: CallbackContext):
    await update.message.reply_text(random.choice(jokes))

# Main bot setup
def main():
    TOKEN = "8074250419:AAFCrIM9t-VmLqQeJZ8rTyHaJv-8evF8mXA"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("joke", joke))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()