import logging
import json
import google.generativeai as genai
from pathlib import Path # ایمپورت Path

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
            # استفاده از Path برای سازگاری بهتر با سیستم‌عامل‌های مختلف
            prompt_path = Path(__file__).parent.parent / 'master_prompt.txt'
            return prompt_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            logger.error("FATAL: فایل master_prompt.txt پیدا نشد.")
            raise

    # --- شروع اصلاحیه: جدا کردن منطق ساخت JSON از مدل ---
    def process_text_to_uks(self, text: str, source: str) -> dict | None:
        """متن خام را با استفاده از پرامپت اصلی به فرمت UKS تبدیل می‌کند."""
        logger.info(f"Processing text from '{source}' to UKS format...")
        
        prompt = self.master_prompt_template.replace(
            "[<<متن خام ورودی از کاربر اینجا قرار می‌گیرد>>]", text
        )

        try:
            response = self.generative_model.generate_content(prompt)
            json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
            
            # مدل فقط بخش‌های تحلیلی را برمی‌گرداند
            analytical_data = json.loads(json_string)
            
            # ما خودمان آبجکت نهایی و معتبر UKS را می‌سازیم
            # این کار از خطاهای JSON جلوگیری می‌کند
            uks_data = analytical_data
            
            # اطمینان از وجود ساختار صحیح و اضافه کردن متن اصلی
            if 'core_content' not in uks_data:
                uks_data['core_content'] = {}
            uks_data['core_content']['original_text'] = text

            if 'source_and_context' not in uks_data:
                 uks_data['source_and_context'] = {}
            uks_data['source_and_context']['source_type'] = source
                 
            logger.info("Successfully generated UKS data.")
            return uks_data
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to process text to UKS. Error: {e}", exc_info=True)
            logger.error(f"LLM Raw Response was: {getattr(response, 'text', 'N/A')}")
            return None
    # --- پایان اصلاحیه ---

    def get_embedding(self, uks_data: dict) -> list | None:
        # ... بقیه کد این تابع بدون تغییر باقی می‌ماند ...
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
