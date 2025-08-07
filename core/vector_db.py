import logging
import uuid
import json  #  ایمپورت کردن کتابخانه json
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self, api_key: str, index_name: str):
        try:
            pc = Pinecone(api_key=api_key)
            self.index_name = index_name
            
            if self.index_name not in pc.list_indexes().names():
                logger.warning(f"ایندکس '{self.index_name}' در Pinecone یافت نشد. در حال ایجاد یک ایندکس جدید...")
                pc.create_index(
                    name=self.index_name,
                    dimension=768, 
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                logger.info(f"ایندکس '{self.index_name}' با موفقیت ایجاد شد.")

            self.pinecone_index = pc.Index(self.index_name)
            logger.info(f"✅ با موفقیت به ایندکس Pinecone '{self.index_name}' متصل شدید.")
            logger.info(self.pinecone_index.describe_index_stats())
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سرویس Pinecone: {e}")
            raise

    def upsert_knowledge(self, uks_data: dict, vector: list) -> str | None:
        """دانش ساختاریافته و بردار آن را در Pinecone ذخیره می‌کند."""
        knowledge_id = str(uuid.uuid4())
        logger.info(f"Preparing to upsert data with ID: {knowledge_id}")
        
        # --- شروع اصلاحیه ---
        # Pinecone مقادیر تودرتو (دیکشنری) را در متادیتا قبول نمی‌کند.
        # ما باید تمام فیلدهایی که دیکشنری یا لیست هستند را به رشته JSON تبدیل کنیم.
        metadata_to_store = {}
        for key, value in uks_data.items():
            if isinstance(value, (dict, list)):
                try:
                    # تبدیل دیکشنری یا لیست به یک رشته متنی با فرمت JSON
                    metadata_to_store[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    # اگر به هر دلیلی قابل تبدیل نبود، آن را به یک رشته ساده تبدیل کن
                    metadata_to_store[key] = str(value)
            else:
                # مقادیر ساده (متن، عدد، بولین) را دست‌نخورده باقی بگذار
                metadata_to_store[key] = value
        # --- پایان اصلاحیه ---

        try:
            self.pinecone_index.upsert(
                vectors=[{'id': knowledge_id, 'values': vector, 'metadata': metadata_to_store}]
            )
            logger.info(f"Successfully upserted data with ID: {knowledge_id} to Pinecone.")
            return knowledge_id
        except Exception as e:
            logger.error(f"Failed to upsert data to Pinecone: {e}", exc_info=True)
            return None
