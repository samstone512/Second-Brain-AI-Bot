import logging
import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger(__name__)

# --- شروع اصلاحیه: تبدیل توابع به حالت همزمان (Synchronous) ---
def convert_voice_to_text(voice_file_path: str) -> str:
    """یک فایل صوتی را به متن تبدیل می‌کند."""
    logger.info("🎵 Converting voice to text...")
    recognizer = sr.Recognizer()
    
    # تعیین فرمت از روی پسوند فایل
    file_extension = Path(voice_file_path).suffix.lower().replace('.', '')
    
    try:
        if file_extension in ['mp3', 'm4a', 'wav', 'ogg']:
             audio = AudioSegment.from_file(voice_file_path, format=file_extension)
        else:
            # اگر فرمت ناشناخته بود، به عنوان ogg امتحان کن
            audio = AudioSegment.from_ogg(voice_file_path)

        wav_path = voice_file_path + ".wav"
        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='fa-IR')
        
        os.remove(wav_path)
        logger.info(f"✅ Voice converted successfully to: '{text}'")
        return text
    except Exception as e:
        logger.error(f"❌ Error converting voice to text: {e}", exc_info=True)
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
        return ""

def extract_text_from_image(image_path: str) -> str:
    """متن را از یک فایل تصویری با استفاده از Gemini استخراج می‌کند."""
    logger.info("🖼️ Extracting text from image...")
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(["Extract all text from this image in Persian.", img])
        extracted_text = response.text
        logger.info(f"✅ Text extracted successfully: '{extracted_text[:100]}...'")
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}", exc_info=True)
        return ""
# --- پایان اصلاحیه ---
