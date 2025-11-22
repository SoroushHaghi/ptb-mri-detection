import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# Load secrets
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
DASHBOARD_URL = "https://ptb-mri-detection.streamlit.app" # Update this after deploying to cloud

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧠 **PTB MRI Assistant**\nPlease send an MRI image for analysis.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo: file_id = update.message.photo[-1].file_id
    elif update.message.document: file_id = update.message.document.file_id
    else: return
    
    await update.message.reply_text(f"✅ Image Received.\n🔬 Analysis Link: {DASHBOARD_URL}/?img_id={file_id}")

if __name__ == '__main__':
    if not TOKEN: print("❌ Error: BOT_TOKEN not found in .env")
    else:
        app = ApplicationBuilder().token(TOKEN).build()
        app.add_handler(CommandHandler('start', start))
        app.add_handler(MessageHandler(filters.PHOTO, handle_image))
        print("🤖 Bot is running securely...")
        app.run_polling()
