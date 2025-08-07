import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import load_secrets
from core.ai_services import AIService
from core.vector_db import VectorDBService
from telegram.handlers import start, handle_text_message, handle_voice_message, handle_photo_message

def main() -> None:
    """Starts the bot and wires up all the services."""
    
    # بارگذاری کلیدها و تنظیمات
    secrets = load_secrets()
    if not all(secrets.values()):
        logging.critical("❌ یکی از کلیدهای API تعریف نشده است. برنامه متوقف شد.")
        return
        
    # راه‌اندازی سرویس‌های اصلی
    try:
        ai_service = AIService(api_key=secrets["GOOGLE_API_KEY"])
        db_service = VectorDBService(api_key=secrets["PINECONE_API_KEY"], index_name=secrets["PINECONE_INDEX_NAME"])
    except Exception as e:
        logging.critical(f"❌ خطا در راه‌اندازی سرویس‌های اصلی: {e}")
        return

    # ساخت اپلیکیشن تلگرام
    builder = Application.builder().token(secrets["TELEGRAM_BOT_TOKEN"])
    application = builder.build()
    
    # به اشتراک‌گذاری نمونه‌های سرویس با context
    application.bot_data["ai_service"] = ai_service
    application.bot_data["db_service"] = db_service
    
    # ثبت هندلرها
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    
    logging.info("🔥 ربات با موفقیت فعال شد! آماده دریافت پیام.")
    # اجرای ربات
    application.run_polling()
