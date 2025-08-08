import logging
from pathlib import Path
import time
from google.api_core.exceptions import TooManyRequests # <-- ایمپورت کردن خطای مشخص

from config import load_secrets
from core.ai_services import AIService
from core.vector_db import VectorDBService
from telegram_bot.utils import convert_voice_to_text, extract_text_from_image

# ... (بخش اول کد بدون تغییر باقی می‌ماند) ...
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
SUPPORTED_AUDIO_EXTENSIONS = ['.ogg', '.mp3', '.wav', '.m4a']
SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md']

def process_file(file_path: Path, ai_service: AIService, db_service: VectorDBService) -> bool:
    logger.info(f"--- Processing file: {file_path.name} ---")
    raw_text = None
    source_type = "Unknown"

    try:
        if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            source_type = "Screenshot"
            raw_text = extract_text_from_image(str(file_path))
        elif file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            source_type = "Audio File"
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
            # این بخش نیازی به False برگرداندن ندارد چون خود تابع بالاتر لاگ خطا را ثبت می‌کند
            return False

        vector = ai_service.get_embedding(uks_data)
        if not vector:
            return False

        knowledge_id = db_service.upsert_knowledge(uks_data, vector)
        if not knowledge_id:
            return False
            
        logger.info(f"✅ Successfully processed and stored {file_path.name} with ID: {knowledge_id}")
        return True
    
    # --- شروع اصلاحیه ---
    except TooManyRequests:
        logger.error(f"RATE LIMIT EXCEEDED while processing {file_path.name}. Skipping this file for now.")
        return False
    # --- پایان اصلاحیه ---
    except Exception as e:
        logger.critical(f"A critical error occurred while processing {file_path.name}: {e}", exc_info=True)
        return False

def run_import(directory_path: str):
    # ... (بخش اول این تابع بدون تغییر باقی می‌ماند) ...
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
    
    files_to_process = [f for f in input_directory.iterdir() if f.is_file()]
    total_files = len(files_to_process)
    
    for i, file_path in enumerate(files_to_process):
        logger.info(f"\n--- Processing file {i+1}/{total_files}: {file_path.name} ---")
        if process_file(file_path, ai_service, db_service):
            success_count += 1
        else:
            failure_count += 1
        
        if i < total_files - 1:
            # --- شروع اصلاحیه ---
            # وقفه را به ۶ ثانیه افزایش می‌دهیم تا مطمئن‌تر باشیم (۱۰ درخواست در دقیقه)
            sleep_duration = 6
            # --- پایان اصلاحیه ---
            logger.info(f"--- Cooling down for {sleep_duration} seconds to respect API rate limits... ---")
            time.sleep(sleep_duration)
    
    logger.info("\n" + "="*50)
    logger.info("Bulk Import Summary")
    logger.info(f"  Successfully processed: {success_count} files")
    logger.info(f"  Failed to process: {failure_count} files")
    logger.info("="*50)
