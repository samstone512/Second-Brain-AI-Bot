import logging
import os
import tempfile
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

# وارد کردن سرویس‌ها و ابزارهای مورد نیاز
from core.ai_services import AIService
from core.vector_db import VectorDBService
from .utils import convert_voice_to_text, extract_text_from_image

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """پیام خوش‌آمدگویی را ارسال می‌کند."""
    user = update.effective_user
    await update.message.reply_html(
        f"سلام {user.mention_html()}!\n\n"
        "من دستیار 'مغز دوم' شما هستم. هر متن، صوت یا تصویری برای من ارسال کنید را پردازش کرده و در پایگاه دانش شما ذخیره می‌کنم.",
    )

async def _process_and_store_text(text: str, source: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """جریان اصلی پردازش متن، تولید UKS، ساخت Embedding و ذخیره‌سازی را اجرا می‌کند."""
    ai_service: AIService = context.bot_data["ai_service"]
    db_service: VectorDBService = context.bot_data["db_service"]
    chat_id = update.message.chat_id

    # 1. Process text to structured JSON (UKS)
    uks_data = ai_service.process_text_to_uks(text, source=source)
    if not uks_data:
        await context.bot.send_message(chat_id, "خطا: نتوانستم متن شما را به فرمت دانش استاندارد تبدیل کنم.")
        return

    # 2. Generate embedding for the knowledge
    vector = ai_service.get_embedding(uks_data)
    if not vector:
        await context.bot.send_message(chat_id, "خطا: نتوانستم بردار معنایی (Embedding) دانش را تولید کنم.")
        return

    # 3. Upsert the data and vector to Pinecone
    knowledge_id = db_service.upsert_knowledge(uks_data, vector)
    if not knowledge_id:
        await context.bot.send_message(chat_id, "خطا: در ذخیره‌سازی دانش در پایگاه داده مشکلی پیش آمد.")
        return

    # 4. Confirm success to the user
    title = uks_data.get("core_content", {}).get("title", "N/A")
    action_type = uks_data.get("actionability", {}).get("actionability_type", "N/A")
    confirmation_message = (
        "✅ دانش با موفقیت در مغز دوم شما ثبت شد!\n\n"
        f"**عنوان:** {title}\n"
        f"**نوع اقدام:** {action_type}\n"
        f"**شناسه:** `{knowledge_id}`"
    )
    await context.bot.send_message(chat_id, confirmation_message, parse_mode=ParseMode.MARKDOWN)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """هندلر برای پیام‌های متنی."""
    raw_text = update.message.text
    logger.info(f"⌨️ Text message received: '{raw_text[:50]}...'")
    await update.message.reply_text("در حال پردازش یادداشت شما...")
    await _process_and_store_text(raw_text, "Telegram Text Message", update, context)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر برای پیام‌های صوتی."""
    logger.info("🎤 Voice message received.")
    await update.message.reply_text("پیام صوتی دریافت شد، در حال تبدیل به متن...")
    
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        await voice_file.download_to_drive(temp_file.name)
        voice_path = temp_file.name
    
    text = await convert_voice_to_text(voice_path)
    os.unlink(voice_path)

    if text:
        await update.message.reply_text(f"📝 متن شناسایی شده: «{text}»\n\nدر حال پردازش و ذخیره‌سازی...")
        await _process_and_store_text(text, "Voice Note", update, context)
    else:
        await update.message.reply_text("❌ متاسفانه نتوانستم صدایتان را تشخیص دهم.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندلر برای پیام‌های تصویری (اسکرین‌شات)."""
    logger.info("🖼️ Photo message received.")
    await update.message.reply_text("تصویر دریافت شد، در حال استخراج متن...")
    
    photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_photo:
        await photo_file.download_to_drive(temp_photo.name)
        photo_path = temp_photo.name
    
    text = await extract_text_from_image(photo_path)
    os.unlink(photo_path)

    if text:
        await update.message.reply_text(f"📝 متن استخراج شده:\n\n«{text}»\n\nدر حال پردازش و ذخیره‌سازی...")
        await _process_and_store_text(text, "Screenshot", update, context)
    else:
        await update.message.reply_text("❌ متنی در تصویر یافت نشد.")
