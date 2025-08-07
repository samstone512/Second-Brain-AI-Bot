import logging
import json
import google.generativeai as genai

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.embedding_model = 'models/text-embedding-004' # جدیدترین مدل امبدینگ
            self.master_prompt_template = self._load_master_prompt()
            logger.info("✅ سرویس هوش مصنوعی (Gemini) با موفقیت راه‌اندازی شد.")
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سرویس Gemini: {e}")
            raise

    def _load_master_prompt(self):
        try:
            with open('master_prompt.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("FATAL: فایل master_prompt.txt پیدا نشد.")
            raise

    def process_text_to_uks(self, text: str, source: str) -> dict | None:
        """متن خام را با استفاده از پرامپت اصلی به فرمت UKS تبدیل می‌کند."""
        logger.info(f"Processing text from '{source}' to UKS format...")
        
        # جایگزینی بخش‌های متغیر در پرامپت
        prompt = self.master_prompt_template.replace(
            "[<<متن خام ورودی از کاربر اینجا قرار می‌گیرد>>]", text
        )

        try:
            response = self.generative_model.generate_content(prompt)
            json_string = response.text.strip().replace('```json', '').replace('```', '').strip()
            uks_data = json.loads(json_string)
            
            # اطمینان از صحت برخی فیلدهای کلیدی
            if 'source_and_context' in uks_data and uks_data['source_and_context'].get('source_type') != source:
                 uks_data['source_and_context']['source_type'] = source # اصلاح نوع منبع
                 
            logger.info("Successfully generated UKS data.")
            return uks_data
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to process text to UKS. Error: {e}", exc_info=True)
            logger.error(f"LLM Raw Response was: {getattr(response, 'text', 'N/A')}")
            return None

    def get_embedding(self, uks_data: dict) -> list | None:
        """برای بخش‌های مهم دانش، یک Embedding تولید می‌کند."""
        
        title = uks_data.get("core_content", {}).get("title", "")
        summary = uks_data.get("core_content", {}).get("summary", "")
        tags = uks_data.get("categorization", {}).get("tags_and_keywords", [])
        
        # ترکیب هوشمندانه متن برای تولید یک Embedding غنی
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
