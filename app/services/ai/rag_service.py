import os
import json
import logging
import asyncio
import tempfile
import time
import vertexai
from vertexai.preview import rag
from google.oauth2 import service_account
from google.auth.transport import requests as google_auth_requests
from google.api_core.exceptions import ResourceExhausted
from pypdf import PdfReader

from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Initialize Vertex AI for RAG with explicit cloud-platform scope.
        # Some environments can otherwise surface invalid_scope during file uploads.
        credentials = self._load_scoped_credentials()
        if credentials is not None:
            vertexai.init(
                project=settings.GCP_PROJECT_ID,
                location=settings.GCP_RAG_LOCATION,
                credentials=credentials,
            )
        else:
            vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_RAG_LOCATION)

        # Small in-process caches to avoid repeated retrieval calls that burn embedding quota.
        self._corpus_cache = {}
        self._grounding_cache = {}
        self._grounding_locks = {}
        self._grounding_ttl_seconds = int(os.getenv("RAG_GROUNDING_CACHE_TTL_SECONDS", "1800"))
        self._grounding_max_queries = int(os.getenv("RAG_GROUNDING_MAX_QUERIES", "1"))
        self._grounding_top_k = int(os.getenv("RAG_GROUNDING_TOP_K", "8"))

    def _load_scoped_credentials(self):
        """Load service-account credentials with explicit cloud-platform scope."""
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or settings.GOOGLE_APPLICATION_CREDENTIALS
        if not creds_path:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Falling back to default ADC resolution.")
            return None

        try:
            return service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception as e:
            logger.warning(f"Failed to load scoped service-account credentials from {creds_path}: {e}")
            return None

    async def _get_corpus_by_topic(self, topic_id: str):
        """Find an existing Corpus by its display name (topic_id)."""
        try:
            cache_key = f"topic_{topic_id}"
            cached = self._corpus_cache.get(cache_key)
            if cached is not None:
                return cached

            corpora = await asyncio.to_thread(rag.list_corpora)
            for c in corpora:
                if c.display_name == f"topic_{topic_id}":
                    self._corpus_cache[cache_key] = c
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
        created = await asyncio.to_thread(rag.create_corpus, display_name=f"topic_{topic_id}")
        self._corpus_cache[f"topic_{topic_id}"] = created
        return created

    async def index_pdf(self, file_path: str, topic_id: str):
        """Upload a PDF file to the Vertex AI RAG Corpus."""
        try:
            logger.info(f"Indexing PDF into Vertex AI RAG: {file_path} for topic: {topic_id}")
            corpus = await self._get_or_create_corpus(topic_id)

            # Prefer explicit text chunk uploads for predictable retrieval granularity.
            text = await asyncio.to_thread(self._extract_pdf_text, file_path)
            chunks = self._chunk_text(text, chunk_chars=1400, overlap_chars=220)

            if not chunks:
                logger.warning("No extractable text found in PDF. Falling back to raw PDF upload.")
                rag_file = await asyncio.to_thread(
                    self._upload_file_scoped,
                    corpus_name=corpus.name,
                    file_path=file_path,
                    display_name=os.path.basename(file_path),
                )
                logger.info(f"Successfully uploaded {file_path} as Vertex RAG File: {rag_file.name}")
                return 1

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            uploaded_count = 0
            for idx, chunk in enumerate(chunks, start=1):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as temp_chunk:
                    temp_chunk.write(chunk)
                    temp_path = temp_chunk.name

                try:
                    display_name = f"{base_name}_chunk_{idx:03d}.txt"
                    await asyncio.to_thread(
                        self._upload_file_scoped,
                        corpus_name=corpus.name,
                        file_path=temp_path,
                        display_name=display_name,
                        mime_type="text/plain",
                    )
                    uploaded_count += 1
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

            logger.info(f"Successfully uploaded {uploaded_count} text chunks for {file_path}")
            return uploaded_count
        except Exception as e:
            logger.error(f"Error indexing PDF {file_path} to Vertex AI: {str(e)}")
            raise e

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract textual content from PDF pages with page headers."""
        reader = PdfReader(file_path)
        sections = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            sections.append(f"[PAGE {idx}]\n{page_text}")
        return "\n\n".join(sections)

    def _chunk_text(self, text: str, chunk_chars: int = 1400, overlap_chars: int = 220):
        """Split text into overlapping chunks while preserving paragraph boundaries."""
        normalized = "\n".join([line.rstrip() for line in text.splitlines()]).strip()
        if not normalized:
            return []

        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= chunk_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(para) <= chunk_chars:
                current = para
            else:
                start = 0
                step = max(chunk_chars - overlap_chars, 200)
                while start < len(para):
                    end = min(start + chunk_chars, len(para))
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end >= len(para):
                        break
                    start += step
                current = ""

        if current:
            chunks.append(current)

        deduped = []
        seen = set()
        for chunk in chunks:
            key = chunk.strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    def _upload_file_scoped(self, corpus_name: str, file_path: str, display_name: str, mime_type: str = "application/pdf"):
        """Upload RagFile with explicit cloud-platform scoped credentials."""
        creds = self._load_scoped_credentials()
        if creds is None:
            raise RuntimeError("Scoped Google credentials are required for RAG file upload.")

        location = settings.GCP_RAG_LOCATION
        upload_request_uri = f"https://{location}-aiplatform.googleapis.com/upload/v1beta1/{corpus_name}/ragFiles:upload"
        headers = {"X-Goog-Upload-Protocol": "multipart"}
        metadata = {"rag_file": {"display_name": display_name}}

        authorized_session = google_auth_requests.AuthorizedSession(credentials=creds)
        with open(file_path, "rb") as file_handle:
            files = {
                "metadata": (None, json.dumps(metadata), "application/json"),
                "file": (os.path.basename(file_path), file_handle, mime_type),
            }
            response = authorized_session.post(
                url=upload_request_uri,
                files=files,
                headers=headers,
            )

        if response.status_code == 404:
            raise ValueError(f"RagCorpus '{corpus_name}' is not found.")

        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Failed in indexing the RagFile due to: {payload['error']}")

        class UploadedRagFile:
            def __init__(self, name: str):
                self.name = name

        return UploadedRagFile(name=payload.get("name", "unknown"))

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
        except ResourceExhausted as e:
            logger.warning(
                f"Vertex embedding quota exhausted while querying topic {topic_id}. "
                f"query='{query[:80]}', top_k={k}. Error: {e}"
            )
            return []
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

    async def get_topic_grounding(self, topic_id: str, limit: int = 20, company_name: str = "") -> str:
        """Fetch representative chunks for AI grounding via semantic search."""
        try:
            normalized_company = company_name.strip()
            cache_key = f"{topic_id}:{normalized_company.lower()}:{limit}"
            now = time.time()
            cached = self._grounding_cache.get(cache_key)
            if cached and cached[0] > now:
                return cached[1]

            lock = self._grounding_locks.setdefault(cache_key, asyncio.Lock())
            async with lock:
                cached = self._grounding_cache.get(cache_key)
                now = time.time()
                if cached and cached[0] > now:
                    return cached[1]

                # Keep retrieval queries minimal because each call can consume embedding quota.
                queries = []
                if normalized_company:
                    queries.append(
                        f"{normalized_company} interview topics, coding rounds, system design and behavioral expectations"
                    )

                queries.append("technical interview question bank and role requirements")

                max_queries = max(1, self._grounding_max_queries)
                effective_limit = max(1, min(limit, self._grounding_top_k))

                results = []
                seen_contents = set()
                for query_text in queries[:max_queries]:
                    query_results = await self.query_topic(topic_id, query=query_text, k=effective_limit)
                    if not query_results:
                        continue

                    for item in query_results:
                        content = (item.get("content") or "").strip()
                        if not content or content in seen_contents:
                            continue
                        seen_contents.add(content)
                        results.append(item)
                        if len(results) >= effective_limit:
                            break

                    if len(results) >= effective_limit:
                        break

                if not results:
                    logger.warning(f"No grounding documents found for topic_{topic_id}")
                    self._grounding_cache[cache_key] = (time.time() + self._grounding_ttl_seconds, "")
                    return ""

                grounding_text = "\n\n".join([f"--- KNOWLEDGE CHUNK ---\n{res['content']}" for res in results])
                self._grounding_cache[cache_key] = (time.time() + self._grounding_ttl_seconds, grounding_text)
                return grounding_text
        except Exception as e:
            logger.error(f"Error getting grounding for topic {topic_id}: {str(e)}")
            return ""

# Singleton instance
rag_service = RAGService()
