import logging
import asyncio
import nest_asyncio

# اعمال پچ برای محیط‌هایی مانند Colab که event loop در حال اجرا دارند
nest_asyncio.apply()

from bot import main as start_bot

# تنظیمات پایه برای لاگ‌گیری
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main():
    """Initializes and runs the bot."""
    logging.info("🤖 شروع به کار دستیار مغز دوم (ساختار ماژولار)...")
    try:
        start_bot()
    except Exception as e:
        logging.critical(f"یک خطای مرگبار در اجرای برنامه رخ داد: {e}", exc_info=True)

if __name__ == "__main__":
    main()
