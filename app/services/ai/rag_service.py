import os
import logging
import asyncio
import vertexai
from vertexai.preview import rag

from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Initialize Vertex AI for RAG using the allowed region (e.g. us-west1)
        # while keeping the main app in your default region if needed.
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_RAG_LOCATION)

    async def _get_corpus_by_topic(self, topic_id: str):
        """Find an existing Corpus by its display name (topic_id)."""
        try:
            corpora = await asyncio.to_thread(rag.list_corpora)
            for c in corpora:
                if c.display_name == f"topic_{topic_id}":
                    return c
        except Exception as e:
            logger.warning(f"Error listing corpora: {e}")
        return None

    async def _get_or_create_corpus(self, topic_id: str):
        """Find the corpus, or create it if it doesn't exist."""
        c = await self._get_corpus_by_topic(topic_id)
        if c:
            return c
        logger.info(f"Creating new Vertex AI RAG Corpus for topic_{topic_id}")
        return await asyncio.to_thread(rag.create_corpus, display_name=f"topic_{topic_id}")

    async def index_pdf(self, file_path: str, topic_id: str):
        """Upload a PDF file to the Vertex AI RAG Corpus."""
        try:
            logger.info(f"Indexing PDF into Vertex AI RAG: {file_path} for topic: {topic_id}")
            corpus = await self._get_or_create_corpus(topic_id)
            
            # Vertex AI RAG natively handles PDF parsing, chunking, and embedding
            rag_file = await asyncio.to_thread(
                rag.upload_file,
                corpus_name=corpus.name,
                path=file_path,
                display_name=os.path.basename(file_path)
            )
            logger.info(f"Successfully uploaded {file_path} as Vertex RAG File: {rag_file.name}")
            return 1 # Managed RAG abstracts chunk counts
        except Exception as e:
            logger.error(f"Error indexing PDF {file_path} to Vertex AI: {str(e)}")
            raise e

    async def query_topic(self, topic_id: str, query: str, k: int = 5):
        """Search for relevant chunks via Vertex AI RAG."""
        try:
            corpus = await self._get_corpus_by_topic(topic_id)
            if not corpus:
                logger.warning(f"No Vertex AI Corpus found for topic_{topic_id}")
                return []
            
            response = await asyncio.to_thread(
                rag.retrieval_query,
                rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
                text=query,
                similarity_top_k=k
            )
            
            results = []
            if getattr(response, 'contexts', None) and getattr(response.contexts, 'contexts', None):
                for context in response.contexts.contexts:
                    results.append({
                        "content": context.text,
                        "metadata": {"source_document": getattr(context, 'source_uri', "Vertex RAG")}
                    })
            return results
        except Exception as e:
            logger.error(f"Error querying Vertex AI topic {topic_id}: {str(e)}")
            return []

    async def delete_topic_collection(self, topic_id: str):
        """Delete an entire Corpus from Vertex AI RAG."""
        try:
            corpus = await self._get_corpus_by_topic(topic_id)
            if corpus:
                logger.info(f"Deleting Vertex AI RAG Corpus: {corpus.name}")
                await asyncio.to_thread(rag.delete_corpus, name=corpus.name)
            return True
        except Exception as e:
            logger.error(f"Error deleting Vertex AI Corpus for topic {topic_id}: {str(e)}")
            return False

    async def get_topic_grounding(self, topic_id: str, limit: int = 20) -> str:
        """Fetch representative chunks for AI grounding via semantic search."""
        try:
            # Since Vertex AI RAG is query-first, we use a general grounding query
            results = await self.query_topic(
                topic_id, 
                query="Key technical requirements, interview questions, and core evaluation concepts", 
                k=limit
            )
            if not results:
                logger.warning(f"No grounding documents found for topic_{topic_id}")
                return ""
            
            grounding_text = "\n\n".join([f"--- KNOWLEDGE CHUNK ---\n{res['content']}" for res in results])
            return grounding_text
        except Exception as e:
            logger.error(f"Error getting grounding for topic {topic_id}: {str(e)}")
            return ""

# Singleton instance
rag_service = RAGService()
