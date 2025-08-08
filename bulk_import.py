import os
import logging
from pathlib import Path

# وارد کردن سرویس‌ها و ابزارهای مورد نیاز از پروژه اصلی
from config import load_secrets
from core.ai_services import AIService
from core.vector_db import VectorDBService
from telegram_bot.utils import convert_voice_to_text, extract_text_from_image

# تنظیمات پایه برای لاگ‌گیری
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# تعریف پسوندهای فایل‌های پشتیبانی شده
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
SUPPORTED_AUDIO_EXTENSIONS = ['.ogg', '.mp3', '.wav', '.m4a']
SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md']


def process_file(file_path: Path, ai_service: AIService, db_service: VectorDBService):
    """یک فایل را پردازش، محتوای آن را استخراج و در دیتابیس ذخیره می‌کند."""
    logger.info(f"--- Processing file: {file_path.name} ---")
    
    raw_text = None
    source_type = "Unknown"

    try:
        if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            source_type = "Screenshot"
            # async/await اینجا کار نمی‌کند، باید تابع را مستقیم صدا بزنیم
            raw_text = extract_text_from_image(str(file_path))
        
        elif file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            source_type = "Audio File"
            # async/await اینجا کار نمی‌کند، باید تابع را مستقیم صدا بزنیم
            raw_text = convert_voice_to_text(str(file_path))

        elif file_path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
            source_type = "Text File"
            raw_text = file_path.read_text(encoding='utf-8')

        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}. Skipping.")
            return False

        if not raw_text or not raw_text.strip():
            logger.error(f"No text could be extracted from {file_path.name}. Skipping.")
            return False

        logger.info(f"Extracted Text: '{raw_text[:100].strip()}...'")

        uks_data = ai_service.process_text_to_uks(raw_text, source=source_type)
        if not uks_data:
            logger.error(f"Failed to convert text to UKS for {file_path.name}.")
            return False

        vector = ai_service.get_embedding(uks_data)
        if not vector:
            logger.error(f"Failed to create embedding for {file_path.name}.")
            return False

        knowledge_id = db_service.upsert_knowledge(uks_data, vector)
        if not knowledge_id:
            logger.error(f"Failed to upsert knowledge for {file_path.name} to Pinecone.")
            return False
            
        logger.info(f"✅ Successfully processed and stored {file_path.name} with ID: {knowledge_id}")
        return True

    except Exception as e:
        logger.critical(f"A critical error occurred while processing {file_path.name}: {e}", exc_info=True)
        return False

# --- شروع اصلاحیه ---
# تابع اصلی را تغییر می‌دهیم تا مسیر را به عنوان آرگومان دریافت کند
def run_import(directory_path: str):
    """نقطه ورود اصلی برای پردازش دسته‌ای که از نوت‌بوک فراخوانی می‌شود."""
    
    input_directory = Path(directory_path)
    if not input_directory.is_dir():
        logger.critical(f"Error: The provided path '{input_directory}' is not a valid directory.")
        return

    logger.info("Loading secrets and initializing services...")
    try:
        secrets = load_secrets()
        ai_service = AIService(api_key=secrets["GOOGLE_API_KEY"])
        db_service = VectorDBService(api_key=secrets["PINECONE_API_KEY"], index_name=secrets["PINECONE_INDEX_NAME"])
    except Exception as e:
        logger.critical(f"Failed to initialize services. Aborting. Error: {e}", exc_info=True)
        return

    logger.info(f"Starting bulk import from directory: '{input_directory}'")
    
    success_count = 0
    failure_count = 0

    for file_path in input_directory.iterdir():
        if file_path.is_file():
            if process_file(file_path, ai_service, db_service):
                success_count += 1
            else:
                failure_count += 1
    
    logger.info("\n" + "="*50)
    logger.info("Bulk Import Summary")
    logger.info(f"  Successfully processed: {success_count} files")
    logger.info(f"  Failed to process: {failure_count} files")
    logger.info("="*50)

# این بخش را حذف می‌کنیم تا فایل به صورت خودکار اجرا نشود
# if __name__ == "__main__":
#     main()
# --- پایان اصلاحیه ---
