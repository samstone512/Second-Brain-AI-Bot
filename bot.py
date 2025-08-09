import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import load_secrets
from core.ai_services import AIService
from core.vector_db import VectorDBService
# ایمپورت کردن هندلرهای جدید و قبلی
from telegram_bot.handlers import (
    start,
    ask_command, # افزوده شده برای فاز ۳
    handle_text_message,
    handle_voice_message,
    handle_photo_message
)

def main() -> None:
    """Starts the bot and wires up all the services."""
    
    secrets = load_secrets()
    if not all(secrets.values()):
        logging.critical("❌ یکی از کلیدهای API تعریف نشده است. برنامه متوقف شد.")
        return
        
    try:
        ai_service = AIService(api_key=secrets["GOOGLE_API_KEY"])
        db_service = VectorDBService(api_key=secrets["PINECONE_API_KEY"], index_name=secrets["PINECONE_INDEX_NAME"])
    except Exception as e:
        logging.critical(f"❌ خطا در راه‌اندازی سرویس‌های اصلی: {e}")
        return

    builder = Application.builder().token(secrets["TELEGRAM_BOT_TOKEN"])
    application = builder.build()
    
    application.bot_data["ai_service"] = ai_service
    application.bot_data["db_service"] = db_service
    
    # --- ثبت هندلرها ---
    application.add_handler(CommandHandler("start", start))
    # --- هندلر جدید برای فاز ۳ ---
    application.add_handler(CommandHandler("ask", ask_command))
    
    # هندلرهای مربوط به ورود داده
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    
    logging.info("🔥 ربات با موفقیت فعال شد! آماده دریافت پیام.")
    application.run_polling()
