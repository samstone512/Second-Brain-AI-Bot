# محتوای جدید برای config.py در شاخه main (نسخه Gemini)
from google.colab import userdata

def load_secrets() -> dict:
    """Loads all necessary secrets from the Colab environment."""
    return {
        "telegram": userdata.get('telegram'),
        "gemini_api_key": userdata.get('GOOGLE_API_KEY'), # <--- تغییر کرد
        "google_creds": userdata.get('Calendar_credentials'),
        "notion_key": userdata.get('NOTION_API_KEY'),
        "notion_ideas_db_id": userdata.get('NOTION_IDEAS_DB_ID'),
        "notion_kb_db_id": userdata.get('NOTION_KB_DB_ID')
    }
