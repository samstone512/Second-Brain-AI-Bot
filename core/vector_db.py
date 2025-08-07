import logging
import uuid
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
                    dimension=768, # ابعاد مدل text-embedding-004
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
        
        # متادیتا شامل تمام داده‌های UKS است برای بازیابی کامل در آینده
        metadata_to_store = uks_data

        try:
            self.pinecone_index.upsert(
                vectors=[{'id': knowledge_id, 'values': vector, 'metadata': metadata_to_store}]
            )
            logger.info(f"Successfully upserted data with ID: {knowledge_id} to Pinecone.")
            return knowledge_id
        except Exception as e:
            logger.error(f"Failed to upsert data to Pinecone: {e}", exc_info=True)
            return None
