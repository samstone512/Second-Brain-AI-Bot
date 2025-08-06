import os
import logging
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import uuid

# ایمپورت‌های لازم
from telegram import Update
from telegram.ext import ContextTypes, Application, MessageHandler, filters
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
import chromadb

logger = logging.getLogger(__name__)

class VoiceAssistantBot:
    def __init__(self, secrets: Dict[str, str]):
        self.secrets = secrets
        self.recognizer = sr.Recognizer()
        self.collection = None

        try:
            genai.configure(api_key=secrets['gemini_api_key'])
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.embedding_model = 'models/embedding-001'
            logging.info("✅ کلاینت Google Gemini با موفقیت راه‌اندازی شد.")
        except Exception as e:
            logging.error(f"❌ خطا در راه‌اندازی کلاینت Gemini: {e}")
            self.gemini_model = None

        # --- بلوک کد با تورفتگی صحیح ---
        try:
            logging.info("☁️ در حال اتصال به دیتابیس ابری ChromaDB...")
            chroma_client = chromadb.CloudClient(
                tenant=secrets['chroma_tenant_id'],
                database='Second Brain',
                api_key=secrets['chroma_api_key']
            )
            self.collection = chroma_client.get_or_create_collection("second_brain_collection")
            logging.info(f"✅ با موفقیت به کالکشن '{self.collection.name}' در ChromaDB Cloud متصل شدید.")
        except Exception as e:
            logging.error(f"❌ خطا در اتصال به ChromaDB Cloud: {e}", exc_info=True)
        # --------------------------------

    def _load_prompt_template(self) -> str:
        try:
            with open("prompt_template.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logging.error("❌ فایل prompt_template.txt یافت نشد!")
            return ""

    async def _process_text_to_uks(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.gemini_model: return None
        
        prompt_template = self._load_prompt_template()
        if not prompt_template: return None

        final_prompt = prompt_template.replace("[<<متن خام ورودی از کاربر اینجا قرار می‌گیرد>>]", text)
        
        logging.info("🤖 در حال پردازش متن به ساختار UKS با Gemini...")
        try:
            response = self.gemini_model.generate_content(final_prompt)
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            logging.info(f"✅ پاسخ JSON از Gemini دریافت شد: {json_text}")
            return json.loads(json_text)
        except Exception as e:
            logging.error(f"❌ خطا در تبدیل متن به UKS: {e}", exc_info=True)
            return None

    async def _add_uks_to_chromadb(self, uks_data: Dict[str, Any]) -> str:
        if not self.collection:
            return "❌ دیتابیس ChromaDB در دسترس نیست."

        try:
            text_to_embed = f"Title: {uks_data['core_content']['title']}\nSummary: {uks_data['core_content']['summary']}"
            
            logging.info(f"🧠 در حال ساخت Embedding برای: '{text_to_embed[:100]}...'")
            embedding_response = genai.embed_content(
                model=self.embedding_model,
                content=text_to_embed,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding_vector = embedding_response['embedding']
            
            metadata = {
                "title": uks_data["core_content"]["title"],
                "summary": uks_data["core_content"]["summary"],
                "original_text": uks_data["core_content"]["original_text"],
                "source_type": uks_data["source_and_context"]["source_type"],
                "source_name": uks_data["source_and_context"].get("source_name") or "Unknown",
                "primary_domain": uks_data["categorization"]["primary_domain"],
                "actionability_type": uks_data["actionability"]["actionability_type"]
            }
            
            doc_id = str(uuid.uuid4())
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding_vector],
                documents=[text_to_embed],
                metadatas=[metadata]
            )
            logging.info(f"✅ دانش با شناسه {doc_id} با موفقیت در ChromaDB ذخیره شد.")
            return f"✅ دانش جدید با عنوان «{metadata['title']}» در مغز دوم شما ذخیره شد."

        except Exception as e:
            logging.error(f"❌ خطا در ذخیره اطلاعات در ChromaDB: {e}", exc_info=True)
            return "مشکلی در ذخیره اطلاعات در پایگاه دانش پیش آمد."

    async def _query_from_chromadb(self, query: str) -> str:
        if not self.collection:
            return "❌ دیتابیس ChromaDB در دسترس نیست."
        
        try:
            logging.info(f"🔎 در حال ساخت Embedding برای پرس‌وجوی: '{query}'")
            query_embedding_response = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            query_vector = query_embedding_response['embedding']

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=5
            )
            
            if not results or not results['documents'][0]:
                return "متاسفانه مطلب مرتبطی در پایگاه دانش شما پیدا نکردم."

            context_str = ""
            for i, metadata in enumerate(results['metadatas'][0]):
                context_str += f"--- سند مرتبط شماره {i+1} ---\n"
                context_str += f"عنوان: {metadata.get('title', 'نامشخص')}\n"
                context_str += f"خلاصه: {metadata.get('summary', 'نامشخص')}\n\n"

            final_prompt = f"شما یک دستیار هوش مصنوعی هستید که بر اساس پایگاه دانش شخصی کاربر به سوالات پاسخ می‌دهید. بر اساس «متن‌های مرتبط» زیر، به «سوال کاربر» یک پاسخ جامع و دقیق به زبان فارسی بدهید.\n\n{context_str}\n\n**سوال کاربر:**\n{query}\n\n**پاسخ شما (به فارسی):**"
            
            logging.info("✍️ در حال تولید پاسخ نهایی با Gemini...")
            final_response = self.gemini_model.generate_content(final_prompt)
            return final_response.text

        except Exception as e:
            logging.error(f"❌ خطا در فرآیند پرس‌وجو از ChromaDB: {e}", exc_info=True)
            return "یک خطای غیرمنتظره در هنگام جستجو رخ داد."

    async def handle_any_input(self, text: str, update: Update):
        await update.message.reply_chat_action('typing')
        uks_data = await self._process_text_to_uks(text)
        
        if not uks_data:
            await update.message.reply_text("❌ مشکلی در تحلیل و درک پیام شما پیش آمد. لطفاً دوباره تلاش کنید.")
            return

        response_text = await self._add_uks_to_chromadb(uks_data)
        await update.message.reply_text(response_text)

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        if user_text.strip().startswith("بپرس:") or user_text.strip().startswith("سوال:"):
            query = user_text.replace("بپرس:", "").replace("سوال:", "").strip()
            logging.info(f"❓ پرس‌وجوی کاربر دریافت شد: '{query}'")
            await update.message.reply_text("🔎 در حال جستجو در مغز دوم شما...")
            answer = await self._query_from_chromadb(query)
            await update.message.reply_text(answer)
        else:
            logging.info(f"⌨️ پیام متنی برای ذخیره دریافت شد: '{user_text}'")
            await self.handle_any_input(user_text, update)

    async def _convert_voice_to_text(self, voice_file_path: str) -> str:
        logging.info("🎵 در حال تبدیل صدا به متن...")
        try:
            audio = AudioSegment.from_ogg(voice_file_path)
            wav_path = voice_file_path + ".wav"
            audio.export(wav_path, format="wav")
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data, language='fa-IR')
            os.remove(wav_path)
            return text
        except Exception as e:
            logging.error(f"❌ خطا در تبدیل صدا به متن: {e}", exc_info=True)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
            return ""

    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🎤 پیام صوتی دریافت شد. لطفاً صبر کنید...")
        voice = update.message.voice
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            voice_file = await context.bot.get_file(voice.file_id)
            await voice_file.download_to_drive(temp_file.name)
            voice_path = temp_file.name
        
        text = await self._convert_voice_to_text(voice_path)
        os.unlink(voice_path)

        if text:
            await update.message.reply_text(f"📝 متن شناسایی شده: «{text}»")
            await self.handle_any_input(text, update)
        else:
            await update.message.reply_text("❌ متاسفانه نتوانستم صدایتان را تشخیص دهم.")

    async def handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🖼️ تصویر دریافت شد. در حال استخراج متن با Gemini...")
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_photo:
            await photo_file.download_to_drive(temp_photo.name)
            photo_path = temp_photo.name
        
        try:
            img = Image.open(photo_path)
            response = self.gemini_model.generate_content(["Extract all text from this image in Persian.", img])
            extracted_text = response.text
            os.unlink(photo_path)
            
            if extracted_text and extracted_text.strip():
                await update.message.reply_text(f"📝 متن استخراج شده:\n\n«{extracted_text}»\n\n📚 در حال افزودن به مغز دوم...")
                await self.handle_any_input(extracted_text, update)
            else:
                await update.message.reply_text("متنی در تصویر یافت نشد.")
        except Exception as e:
            logging.error(f"❌ خطا در پردازش تصویر: {e}", exc_info=True)
            await update.message.reply_text("مشکلی در پردازش تصویر پیش آمد.")
            if os.path.exists(photo_path):
                os.unlink(photo_path)

    async def run(self):
        logging.info("\n🚀 در حال راه‌اندازی ربات تلگرام...")
        app = Application.builder().token(self.secrets['telegram']).build()
        
        app.add_handler(MessageHandler(filters.VOICE, self.handle_voice_message))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo_message))
        
        logging.info("🔥 ربات با موفقیت فعال شد! آماده دریافت پیام.")
        await app.run_polling()
