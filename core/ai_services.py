import logging
import json
import google.generativeai as genai
from pathlib import Path
import re # <-- ۱. کتابخانه عبارات باقاعده را وارد می‌کنیم

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.embedding_model = 'models/text-embedding-004'
            self.master_prompt_template = self._load_master_prompt()
            logger.info("✅ سرویس هوش مصنوعی (Gemini) با موفقیت راه‌اندازی شد.")
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سرویس Gemini: {e}")
            raise

    def _load_master_prompt(self):
        try:
            prompt_path = Path(__file__).parent.parent / 'master_prompt.txt'
            return prompt_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            logger.error("FATAL: فایل master_prompt.txt پیدا نشد.")
            raise

    def process_text_to_uks(self, text: str, source: str) -> dict | None:
        """متن خام را با استفاده از پرامپت اصلی به فرمت UKS تبدیل می‌کند."""
        logger.info(f"Processing text from '{source}' to UKS format...")
        
        prompt = self.master_prompt_template.replace(
            "[<<متن خام ورودی از کاربر اینجا قرار می‌گیرد>>]", text
        )

        try:
            response = self.generative_model.generate_content(prompt)
            raw_response_text = response.text
            
            # --- شروع اصلاحیه: استخراج هوشمند JSON ---
            # ۲. از عبارات باقاعده برای پیدا کردن بلوک JSON استفاده می‌کنیم
            # این الگو به دنبال هر چیزی می‌گردد که بین ```json و ``` یا بین { و } قرار دارد
            match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', raw_response_text, re.DOTALL)
            
            if not match:
                logger.error("Could not find a valid JSON block in the LLM's response.")
                logger.error(f"LLM Raw Response was: {raw_response_text}")
                return None
            
            # گروهی که پیدا شده را به عنوان رشته JSON استخراج می‌کنیم
            json_string = match.group(1) or match.group(2)
            # --- پایان اصلاحیه ---

            analytical_data = json.loads(json_string)
            
            uks_data = analytical_data
            
            if 'core_content' not in uks_data:
                uks_data['core_content'] = {}
            uks_data['core_content']['original_text'] = text

            if 'source_and_context' not in uks_data:
                 uks_data['source_and_context'] = {}
            uks_data['source_and_context']['source_type'] = source
                 
            logger.info("Successfully generated UKS data.")
            return uks_data
        except (json.JSONDecodeError, Exception) as e:
            # حالا که پاسخ خام را قبل از هر کاری ذخیره کرده‌ایم، لاگ دقیق‌تری خواهیم داشت
            raw_response_for_log = locals().get('raw_response_text', 'Response not captured')
            logger.error(f"Failed to process text to UKS for file. Error: {e}", exc_info=True)
            logger.error(f"LLM Raw Response was: {raw_response_for_log}")
            return None

    def get_embedding(self, uks_data: dict) -> list | None:
        """برای بخش‌های مهم دانش، یک Embedding تولید می‌کند."""
        title = uks_data.get("core_content", {}).get("title", "")
        summary = uks_data.get("core_content", {}).get("summary", "")
        tags = uks_data.get("categorization", {}).get("tags_and_keywords", [])
        
        text_to_embed = f"Title: {title}\nSummary: {summary}\nTags: {', '.join(tags)}"
        logger.info(f"Generating embedding for: '{text_to_embed[:100]}...'")

        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text_to_embed,
                task_type="RETRIEVAL_DOCUMENT"
            )
            logger.info("Successfully generated embedding.")
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            return None
 

