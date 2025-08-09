import logging
import json
import google.generativeai as genai
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.embedding_model = 'models/text-embedding-004'
            self.master_prompt_template = self._load_prompt_template('master_prompt.txt')
            # --- افزوده شده برای فاز ۳ ---
            self.rag_prompt_template = self._load_prompt_template('rag_prompt.txt')
            logger.info("✅ سرویس هوش مصنوعی (Gemini) با موفقیت راه‌اندازی شد.")
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سرویس Gemini: {e}")
            raise

    def _load_prompt_template(self, file_name: str) -> str:
        """یک فایل پرامپت را از ریشه پروژه بارگذاری می‌کند."""
        try:
            prompt_path = Path(__file__).parent.parent / file_name
            return prompt_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            logger.error(f"FATAL: فایل پرامپت '{file_name}' پیدا نشد.")
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
            
            match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', raw_response_text, re.DOTALL)
            
            if not match:
                logger.error("Could not find a valid JSON block in the LLM's response.")
                logger.error(f"LLM Raw Response was: {raw_response_text}")
                return None
            
            json_string = match.group(1) or match.group(2)
            
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
            raw_response_for_log = locals().get('raw_response_text', 'Response not captured')
            logger.error(f"Failed to process text to UKS for file. Error: {e}", exc_info=True)
            logger.error(f"LLM Raw Response was: {raw_response_for_log}")
            return None

    def get_document_embedding(self, uks_data: dict) -> list | None:
        """برای یک دانش ساختاریافته، یک Embedding از نوع DOCUMENT تولید می‌کند."""
        title = uks_data.get("core_content", {}).get("title", "")
        summary = uks_data.get("core_content", {}).get("summary", "")
        tags = uks_data.get("categorization", {}).get("tags_and_keywords", [])
        
        text_to_embed = f"Title: {title}\nSummary: {summary}\nTags: {', '.join(tags)}"
        logger.info(f"Generating DOCUMENT embedding for: '{text_to_embed[:100]}...'")

        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text_to_embed,
                task_type="RETRIEVAL_DOCUMENT"
            )
            logger.info("Successfully generated document embedding.")
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate document embedding: {e}", exc_info=True)
            return None

    # --- متد جدید برای فاز ۳ ---
    def get_query_embedding(self, query: str) -> list | None:
        """برای یک سوال (query)، یک Embedding از نوع QUERY تولید می‌کند."""
        logger.info(f"Generating QUERY embedding for: '{query}'")
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            logger.info("Successfully generated query embedding.")
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
            return None

    # --- متد جدید برای فاز ۳ ---
    def generate_rag_response(self, query: str, context: str) -> str:
        """بر اساس سوال کاربر و کانتکست یافت‌شده، پاسخ نهایی را تولید می‌کند."""
        logger.info("Generating RAG response...")
        prompt = self.rag_prompt_template.format(context=context, user_query=query)
        
        try:
            response = self.generative_model.generate_content(prompt)
            logger.info("Successfully generated RAG response.")
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}", exc_info=True)
            return "متاسفانه در هنگام تولید پاسخ خطایی رخ داد. لطفاً دوباره تلاش کنید."
