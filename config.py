import os
from dotenv import load_dotenv
from google.colab import userdata

def load_secrets() -> dict:
    """Loads all necessary secrets from .env file or Colab environment."""
    load_dotenv() # برای توسعه محلی می‌توانید از یک فایل .env استفاده کنید
    
    # اولویت با Colab Secrets است
    secrets = {
        "TELEGRAM_BOT_TOKEN": userdata.get('TELEGRAM_BOT_TOKEN'),
        "GOOGLE_API_KEY": userdata.get('GOOGLE_API_KEY'),
        "PINECONE_API_KEY": userdata.get('PINECONE_API_KEY'),
        "PINECONE_INDEX_NAME": userdata.get('PINECONE_INDEX_NAME', 'second-brain-index')
    }
    
    # اگر در Colab نبود، از متغیرهای محیطی سیستم بخوان
    for key, value in secrets.items():
        if value is None:
            secrets[key] = os.getenv(key)
            
    return secrets
