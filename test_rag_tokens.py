"""
RAG token usage diagnostic.
Creates a small synthetic knowledge base, indexes it, then
measures embedding API calls and the size of grounding text injected
into the LLM prompt.

Run:
    cd /path/to/ai
    source venv/bin/activate
    GOOGLE_API_KEY=<key> python3 test_rag_tokens.py
"""

import asyncio
import os
import sys
import time
import pathlib
import tempfile

# ── Allow importing app modules without a running server ────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.environ.setdefault("JWT_ACCESS_SECRET", "test")
os.environ.setdefault("DEEPGRAM_KEY", "test")

# ── Synthetic interview transcript (simulates an admin upload) ───────────────
SAMPLE_TRANSCRIPT = """
Google Software Engineer – Coding Interview Transcript

Interviewer: Hi, thanks for joining. Let's start with a warm-up.
Can you explain the difference between a stack and a queue?

Candidate: Sure. A stack follows LIFO — last in, first out.
A queue follows FIFO — first in, first out.

---

Q1: Reverse a linked list.
Given the head of a singly linked list, reverse it and return the reversed list.
Example: 1->2->3->4->5 becomes 5->4->3->2->1.
Expected: O(n) time, O(1) space using three-pointer iteration.

---

Q2: Two Sum.
Given an integer array nums and a target, return the indices of the two numbers
that add up to target. You may assume exactly one solution exists.
Expected: O(n) using a hash map.

---

Q3: Design a URL shortener.
Walk through the high-level architecture.
Topics: hashing, collision resolution, database schema, read-heavy caching.

---

Q4: Longest Substring Without Repeating Characters.
Given a string s, find the length of the longest substring without repeating characters.
Expected: O(n) sliding window with a set.

---

Q5: System Design – Design Twitter's feed.
Cover: fanout on write vs fanout on read, Redis caching, sharding strategy,
handling celebrity accounts, eventual consistency trade-offs.

---

Q6: Binary Search in rotated sorted array.
Given a sorted array rotated at an unknown pivot, search for a target in O(log n).
"""


# ── Word / token counter (rough: 1 token ≈ 0.75 words) ─────────────────────
def word_count(text: str) -> int:
    return len(text.split())

def approx_tokens(text: str) -> int:
    return int(word_count(text) / 0.75)


# ── Embedding call counter ───────────────────────────────────────────────────
_embed_calls = 0
_embed_texts_total = 0

_real_embed_texts = None
_real_embed_query = None


def patched_embed_texts(self, texts):
    global _embed_calls, _embed_texts_total
    _embed_calls += len(texts)          # one API call per text
    _embed_texts_total += sum(approx_tokens(t) for t in texts)
    return _real_embed_texts(self, texts)


def patched_embed_query(self, query):
    global _embed_calls, _embed_texts_total
    _embed_calls += 1
    _embed_texts_total += approx_tokens(query)
    return _real_embed_query(self, query)


# ── Main test ────────────────────────────────────────────────────────────────
async def main():
    # Import after path setup
    from app.services.ai.rag_service import RAGService

    # Patch to count calls
    global _real_embed_texts, _real_embed_query
    _real_embed_texts = RAGService._embed_texts
    _real_embed_query = RAGService._embed_query
    RAGService._embed_texts = patched_embed_texts
    RAGService._embed_query = patched_embed_query

    svc = RAGService()
    topic_id = "test_diagnostic_topic"

    # ── Write transcript to a temp file ────────────────────────────────────
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp_path = f.name

    print("=" * 60)
    print("RAG TOKEN USAGE DIAGNOSTIC")
    print("=" * 60)
    print(f"\n[Transcript] {word_count(SAMPLE_TRANSCRIPT)} words "
          f"(~{approx_tokens(SAMPLE_TRANSCRIPT)} tokens)\n")

    # ── INDEXING ────────────────────────────────────────────────────────────
    print("── INDEXING ──")
    calls_before = _embed_calls
    tokens_before = _embed_texts_total
    t0 = time.time()

    chunk_count = await svc.index_pdf(tmp_path, topic_id)

    index_time = time.time() - t0
    index_calls = _embed_calls - calls_before
    index_tokens = _embed_texts_total - tokens_before

    print(f"  Chunks created      : {chunk_count}")
    print(f"  Embedding API calls : {index_calls}  (1 call per chunk)")
    print(f"  Tokens sent to embed: ~{index_tokens}")
    print(f"  Time                : {index_time:.1f}s")

    # ── GROUNDING (simulates what happens at interview start) ────────────────
    print("\n── GROUNDING (each interview start) ──")
    calls_before = _embed_calls
    tokens_before = _embed_texts_total
    t0 = time.time()

    grounding = await svc.get_topic_grounding(topic_id, limit=20, company_name="Google")

    ground_time = time.time() - t0
    ground_calls = _embed_calls - calls_before
    ground_tokens_embed = _embed_texts_total - tokens_before
    grounding_prompt_tokens = approx_tokens(grounding)

    print(f"  Queries fired       : {ground_calls}  (embedding API calls)")
    print(f"  Tokens sent to embed: ~{ground_tokens_embed}")
    print(f"  Grounding text size : {len(grounding)} chars / "
          f"~{grounding_prompt_tokens} tokens injected into LLM prompt")
    print(f"  Time                : {ground_time:.1f}s")

    # ── CACHED GROUNDING ────────────────────────────────────────────────────
    print("\n── CACHED GROUNDING (subsequent calls) ──")
    calls_before = _embed_calls
    t0 = time.time()
    _ = await svc.get_topic_grounding(topic_id, limit=20, company_name="Google")
    print(f"  Embedding API calls : {_embed_calls - calls_before}  (should be 0 — served from cache)")
    print(f"  Time                : {time.time() - t0:.4f}s")

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    print("\n── SUMMARY ──")
    print(f"  Total embedding API calls : {_embed_calls}")
    print(f"  Total tokens to embeddings: ~{_embed_texts_total}")
    print(f"  LLM prompt overhead/session: ~{grounding_prompt_tokens} tokens")
    print()

    if grounding_prompt_tokens > 3000:
        print("⚠  WARNING: Grounding text is large (>3000 tokens injected per session).")
        print("   Consider reducing RAG_GROUNDING_TOP_K or trimming chunk size.")
    else:
        print("✓  Grounding size looks reasonable.")

    if ground_calls > 2:
        print(f"⚠  WARNING: {ground_calls} embedding calls per session — see optimization notes below.")
    else:
        print(f"✓  Embedding calls per grounding: {ground_calls}")

    # ── CLEANUP ─────────────────────────────────────────────────────────────
    os.remove(tmp_path)
    try:
        svc._chroma.delete_collection(f"topic_{topic_id}")
    except Exception:
        pass

    print("\n[Cleanup] Temp collection deleted.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
