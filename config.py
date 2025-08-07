import logging
from google.colab import userdata

def load_secrets() -> dict:
    """
    تمام کلیدهای لازم را به صورت امن از Google Colab Secrets می‌خواند.
    """
    logging.info("در حال خواندن کلیدهای محرمانه از Colab Secrets...")
    
    # تعریف نام کلیدهایی که در Colab Secrets ذخیره کرده‌اید
    required_secrets = [
        "telegram", 
        "GOOGLE_API_KEY", 
        "PINECONE_API_KEY", 
        "PINECONE_INDEX_NAME"
    ]
    
    secrets = {}
    all_found = True
    
    for secret_name in required_secrets:
        value = userdata.get(secret_name)
        if value is None:
            logging.error(f"❌ کلید محرمانه '{secret_name}' در Colab Secrets یافت نشد یا فعال نیست!")
            all_found = False
        secrets[secret_name] = value
        
    if not all_found:
        raise ValueError("یک یا چند کلید محرمانه ضروری یافت نشد. لطفاً تنظیمات Colab Secrets را بررسی کنید.")

    # نگاشت نام‌های Colab به نام‌های مورد استفاده در برنامه
    # این کار باعث می‌شود بقیه کد نیازی به دانستن نام دقیق کلیدها در Colab نداشته باشد
    return {
        "TELEGRAM_BOT_TOKEN": secrets["telegram"],
        "GOOGLE_API_KEY": secrets["GOOGLE_API_KEY"],
        "PINECONE_API_KEY": secrets["PINECONE_API_KEY"],
        "PINECONE_INDEX_NAME": secrets["PINECONE_INDEX_NAME"]
    }
