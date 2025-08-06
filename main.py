import asyncio
import logging
from config import load_secrets
from bot import VoiceAssistantBot

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start():
    """Initializes and runs the bot."""
    logging.info("🤖 شروع به کار دستیار شخصی هوشمند (از پروژه ماژولار)...")
    try:
        secrets = load_secrets()
        # --- اصلاحیه اصلی اینجاست ---
        if not all([secrets["telegram"], secrets["gemini_api_key"], secrets["notion_key"]]):
            logging.critical("❌ خطا: کلیدهای اصلی (تلگرام، جمینی، نوشن) تعریف نشده‌اند.")
            return

        bot = VoiceAssistantBot(secrets)
        
        # این بخش دیگر ضروری نیست چون از نوشن به عنوان دیتابیس استفاده نمی‌کنیم
        # if secrets.get('notion_ideas_db_id'):
        #     bot._discover_notion_db_properties(secrets['notion_ideas_db_id'])
        # if secrets.get('notion_kb_db_id'):
        #     bot._discover_notion_db_properties(secrets['notion_kb_db_id'])
        
        # تابع setup_google_calendar همچنان برای تقویم لازم است
        #if bot.setup_google_calendar():
        await bot.run()
            
    except Exception as e:
        logging.critical(f"یک خطای مرگبار در اجرای برنامه رخ داد: {e}", exc_info=True)

#if __name__ == "__main__":
    #import nest_asyncio
    #nest_asyncio.apply()
    #asyncio.run(start())
