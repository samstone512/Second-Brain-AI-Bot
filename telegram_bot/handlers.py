import logging
import os
import tempfile
from telegram import Update, PhotoSize
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from core.ai_services import AIService
from core.vector_db import VectorDBService
from .utils import convert_voice_to_text, extract_text_from_image

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        f"سلام {user.mention_html()}!\n\n"
        "من دستیار 'مغز دوم' شما هستم. هر متن، صوت یا تصویری برای من ارسال کنید را پردازش کرده و در پایگاه دانش شما ذخیره می‌کنم.",
    )

async def _process_and_store_text(text: str, source: str, update: Update, context: ContextTypes.DEFAULT_TYPE, reply_to_message_id: int):
    ai_service: AIService = context.bot_data["ai_service"]
    db_service: VectorDBService = context.bot_data["db_service"]
    chat_id = update.message.chat_id

    try:
        uks_data = ai_service.process_text_to_uks(text, source=source)
        if not uks_data:
            await context.bot.send_message(chat_id, "❌ خطا: نتوانستم متن شما را به فرمت دانش استاندارد تبدیل کنم.", reply_to_message_id=reply_to_message_id)
            return

        vector = ai_service.get_embedding(uks_data)
        if not vector:
            await context.bot.send_message(chat_id, "❌ خطا: نتوانستم بردار معنایی (Embedding) دانش را تولید کنم.", reply_to_message_id=reply_to_message_id)
            return

        knowledge_id = db_service.upsert_knowledge(uks_data, vector)
        if not knowledge_id:
            await context.bot.send_message(chat_id, "❌ خطا: در ذخیره‌سازی دانش در پایگاه داده مشکلی پیش آمد.", reply_to_message_id=reply_to_message_id)
            return

        title = uks_data.get("core_content", {}).get("title", "N/A")
        confirmation_message = (
            f"✅ دانش با موفقیت ثبت شد!\n\n"
            f"**عنوان:** {title}\n"
            f"**شناسه:** `{knowledge_id}`"
        )
        await context.bot.send_message(chat_id, confirmation_message, parse_mode=ParseMode.MARKDOWN, reply_to_message_id=reply_to_message_id)
    except Exception as e:
        logger.error(f"An unexpected error occurred in _process_and_store_text: {e}", exc_info=True)
        await context.bot.send_message(chat_id, f"❌ یک خطای غیرمنتظره رخ داد: {e}", reply_to_message_id=reply_to_message_id)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw_text = update.message.text
    logger.info(f"⌨️ Text message received: '{raw_text[:50]}...'")
    # Reply to the specific message being processed
    await _process_and_store_text(raw_text, "Telegram Text Message", update, context, reply_to_message_id=update.message.message_id)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("🎤 Voice message received.")
    
    # Reply to the voice message to indicate start of processing
    processing_message = await update.message.reply_text("در حال تبدیل پیام صوتی به متن...")
    
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        await voice_file.download_to_drive(temp_file.name)
        voice_path = temp_file.name
    
    text = convert_voice_to_text(voice_path)
    os.unlink(voice_path)

    if text:
        await processing_message.edit_text(f"📝 متن شناسایی شده: «{text}»\n\nدر حال پردازش و ذخیره‌سازی...")
        await _process_and_store_text(text, "Voice Note", update, context, reply_to_message_id=update.message.message_id)
    else:
        await processing_message.edit_text("❌ متاسفانه نتوانستم صدایتان را تشخیص دهم.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles both single photos and albums (groups of photos)."""
    logger.info("🖼️ Photo message received.")
    
    # For albums, media_group_id is present. We process only the first message of an album.
    if update.message.media_group_id and context.chat_data.get(update.message.media_group_id):
        return
    if update.message.media_group_id:
        context.chat_data[update.message.media_group_id] = True

    # Process each photo in the message (usually one, but can be more in an album)
    photos = update.message.photo
    if not photos:
        # This can happen if the message is part of an album but has no photo itself (e.g., a caption)
        return

    # Use the highest resolution photo
    photo_to_process: PhotoSize = photos[-1]
    
    processing_message = await update.message.reply_text(f"در حال استخراج متن از تصویر...")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_photo:
        photo_file = await context.bot.get_file(photo_to_process.file_id)
        await photo_file.download_to_drive(temp_photo.name)
        photo_path = temp_photo.name
    
    text = extract_text_from_image(photo_path)
    os.unlink(photo_path)

    if text:
        await processing_message.edit_text(f"📝 متن استخراج شده:\n\n«{text}»\n\nدر حال پردازش و ذخیره‌سازی...")
        await _process_and_store_text(text, "Screenshot", update, context, reply_to_message_id=update.message.message_id)
    else:
        await processing_message.edit_text("❌ متنی در تصویر یافت نشد.")
