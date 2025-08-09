# telegram_bot/utils.py

import logging
import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import google.generativeai as genai
from pathlib import Path

logger = logging.getLogger(__name__)

def convert_voice_to_text(voice_file_path: str) -> str:
    """یک فایل صوتی را به متن تبدیل می‌کند. (نسخه همزمان)"""
    logger.info(f"🎵 Converting voice from: {voice_file_path}")
    recognizer = sr.Recognizer()
    
    file_extension = Path(voice_file_path).suffix.lower().replace('.', '')
    wav_path = voice_file_path + ".wav"
    
    try:
        if file_extension in ['mp3', 'm4a', 'wav', 'ogg']:
             audio = AudioSegment.from_file(voice_file_path, format=file_extension)
        else:
            audio = AudioSegment.from_ogg(voice_file_path)

        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='fa-IR')
        
        logger.info(f"✅ Voice converted successfully to: '{text}'")
        return text
    except Exception as e:
        logger.error(f"❌ Error converting voice to text: {e}", exc_info=True)
        return ""
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

def extract_text_from_image(image_path: str) -> str:
    """متن را از یک فایل تصویری استخراج می‌کند. (نسخه همزمان و چندزبانه)"""
    logger.info(f"🖼️ Extracting text from image: {image_path}")
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # --- شروع اصلاحیه ---
        # دستور را به یک دستور عمومی برای استخراج تمام متون تغییر می‌دهیم
        response = model.generate_content(["Extract all text from this image.", img])
        # --- پایان اصلاحیه ---

        extracted_text = response.text
        logger.info(f"✅ Text extracted successfully: '{extracted_text[:100]}...'")
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}", exc_info=True)
        return ""
