import logging
import uuid
import json
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
        
        metadata_to_store = {}
        for key, value in uks_data.items():
            if isinstance(value, (dict, list)):
                try:
                    metadata_to_store[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    metadata_to_store[key] = str(value)
            else:
                metadata_to_store[key] = value

        try:
            self.pinecone_index.upsert(
                vectors=[{'id': knowledge_id, 'values': vector, 'metadata': metadata_to_store}]
            )
            logger.info(f"Successfully upserted data with ID: {knowledge_id} to Pinecone.")
            return knowledge_id
        except Exception as e:
            logger.error(f"Failed to upsert data to Pinecone: {e}", exc_info=True)
            return None

    # --- متد جدید برای فاز ۳ ---
    def search(self, vector: list, top_k: int = 5) -> list[dict]:
        """دانش‌های مرتبط را بر اساس یک بردار جستجو می‌کند."""
        logger.info(f"Searching for top {top_k} similar documents.")
        if not vector:
            logger.warning("Search called with an empty vector.")
            return []
            
        try:
            results = self.pinecone_index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )
            
            processed_matches = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                # بازگرداندن فیلدهای JSON به حالت اولیه (دیکشنری)
                for key, value in metadata.items():
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            metadata[key] = json.loads(value)
                        except json.JSONDecodeError:
                            # اگر تبدیل ناموفق بود، همان رشته باقی بماند
                            pass
                processed_matches.append(metadata)

            logger.info(f"Found {len(processed_matches)} relevant documents.")
            return processed_matches
        except Exception as e:
            logger.error(f"An error occurred during Pinecone search: {e}", exc_info=True)
            return []
