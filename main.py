import logging
import asyncio
import nest_asyncio

# --- تغییر اصلی اینجاست: تنظیمات لاگ‌گیری در بالاترین سطح برنامه ---
# این کار تضمین می‌کند که تمام لاگ‌های ما در خروجی Colab نمایش داده شوند.
#logging.basicConfig(
#    level=logging.INFO,
#    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#    datefmt='%Y-%m-%d %H:%M:%S'
#)
# --- پایان تغییر ---

# اعمال پچ برای محیط‌هایی مانند Colab که event loop در حال اجرا دارند
nest_asyncio.apply()

from bot import main as start_bot

def main():
    """Initializes and runs the bot."""
    logging.info("🤖 شروع به کار دستیار مغز دوم (ساختار ماژولار)...")
    try:
        start_bot()
    except Exception as e:
        logging.critical(f"یک خطای مرگبار در اجرای برنامه رخ داد: {e}", exc_info=True)

if __name__ == "__main__":
    main()
