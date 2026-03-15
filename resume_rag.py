"""
Resume RAG System - Document Processing Pipeline
================================================
Loads PDF resumes, chunks them by section, extracts metadata,
generates embeddings via Gemini, and stores in ChromaDB.
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Optional

import pdfplumber
import tiktoken
import chromadb
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "gemini-embedding-2-preview"
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHROMA_COLLECTION = "resumes"
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
RESUME_DIR = os.path.join(os.path.dirname(__file__), "diverse_resume")

SECTION_HEADINGS = [
    "summary", "objective", "profile", "about",
    "experience", "work experience", "professional experience", "employment",
    "education", "academic", "qualifications",
    "skills", "technical skills", "core competencies", "technologies",
    "projects", "personal projects", "key projects",
    "certifications", "certificates", "licenses",
    "achievements", "awards", "honors",
    "publications", "research",
    "interests", "hobbies",
    "references",
    "contact", "personal information",
]

# Build a regex that matches section headings (case-insensitive, at line start)
_heading_pattern = "|".join(re.escape(h) for h in SECTION_HEADINGS)
SECTION_RE = re.compile(
    rf"^[\s]*(?P<heading>{_heading_pattern})[\s]*[:\-–—]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _sliding_window_chunks(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by token count."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------
class PDFLoader:
    """Load PDF files and extract text using pdfplumber."""

    @staticmethod
    def load_single(pdf_path: str) -> str:
        """Extract all text from a single PDF."""
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    @staticmethod
    def load_all(directory: str = RESUME_DIR) -> list[dict]:
        """
        Load all PDFs from a directory.
        Returns list of {"path": ..., "filename": ..., "text": ...}
        """
        results = []
        pdf_dir = Path(directory)
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            try:
                text = PDFLoader.load_single(str(pdf_file))
                if text.strip():
                    results.append({
                        "path": str(pdf_file),
                        "filename": pdf_file.name,
                        "text": text,
                    })
                    print(f"  ✓ Loaded: {pdf_file.name}")
                else:
                    print(f"  ⚠ Empty text: {pdf_file.name}")
            except Exception as e:
                print(f"  ✗ Failed: {pdf_file.name} → {e}")
        print(f"\nLoaded {len(results)} resumes from {directory}")
        return results


# ---------------------------------------------------------------------------
# Resume Chunker  (section-aware)
# ---------------------------------------------------------------------------
class ResumeChunker:
    """
    Splits resume text into chunks, preserving section boundaries.
    Large sections are further split with a sliding window.
    """

    def __init__(self, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, text: str) -> list[dict]:
        """
        Returns list of {"text": ..., "section": ...}
        """
        sections = self._split_into_sections(text)
        chunks = []
        for section_name, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue
            if count_tokens(section_text) <= self.max_tokens:
                chunks.append({"text": section_text, "section": section_name})
            else:
                sub_chunks = _sliding_window_chunks(
                    section_text, self.max_tokens, self.overlap
                )
                for sc in sub_chunks:
                    chunks.append({"text": sc, "section": section_name})
        return chunks

    @staticmethod
    def _split_into_sections(text: str) -> list[tuple[str, str]]:
        """Split text by detected section headings."""
        matches = list(SECTION_RE.finditer(text))
        if not matches:
            return [("full_resume", text)]

        sections = []
        # Text before first heading
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("header", preamble))

        for i, match in enumerate(matches):
            heading = match.group("heading").strip().lower()
            # Normalize heading
            heading = ResumeChunker._normalize_heading(heading)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))

        return sections

    @staticmethod
    def _normalize_heading(heading: str) -> str:
        """Map variations to canonical section names."""
        heading = heading.lower().strip()
        mapping = {
            "work experience": "experience",
            "professional experience": "experience",
            "employment": "experience",
            "academic": "education",
            "qualifications": "education",
            "technical skills": "skills",
            "core competencies": "skills",
            "technologies": "skills",
            "personal projects": "projects",
            "key projects": "projects",
            "certificates": "certifications",
            "licenses": "certifications",
            "honors": "achievements",
            "awards": "achievements",
            "objective": "summary",
            "profile": "summary",
            "about": "summary",
            "personal information": "contact",
            "hobbies": "interests",
        }
        return mapping.get(heading, heading)


# ---------------------------------------------------------------------------
# Metadata Extractor
# ---------------------------------------------------------------------------
class MetadataExtractor:
    """Extract structured metadata from raw resume text."""

    @staticmethod
    def extract(text: str) -> dict:
        return {
            "candidate_name": MetadataExtractor._extract_name(text),
            "skills": MetadataExtractor._extract_skills(text),
            "experience_years": MetadataExtractor._extract_experience_years(text),
            "education": MetadataExtractor._extract_education(text),
        }

    @staticmethod
    def _extract_name(text: str) -> str:
        """Heuristic: first non-empty line is usually the candidate name."""
        lines = text.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            # Skip lines that look like contact info / emails / phones
            if re.search(r"[@\d{5,}]", line):
                continue
            if line and len(line) < 60 and not line.startswith(("http", "www")):
                # Remove common prefixes
                cleaned = re.sub(r"^(name\s*[:–-]\s*)", "", line, flags=re.IGNORECASE)
                return cleaned.strip()
        return "Unknown"

    @staticmethod
    def _extract_skills(text: str) -> str:
        """Extract skills from a Skills/Technical Skills section."""
        # Try to find a skills section
        skills_match = re.search(
            r"(?:skills|technical skills|core competencies|technologies)\s*[:\-–—]?\s*\n(.*?)(?=\n\s*(?:"
            + _heading_pattern
            + r")\s*[:\-–—]?\s*$|\Z)",
            text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if skills_match:
            skills_text = skills_match.group(1).strip()
            # Clean up: flatten to comma-separated
            skills_text = re.sub(r"[\n•·▪►●■\-]+", ", ", skills_text)
            skills_text = re.sub(r"\s*,\s*,\s*", ", ", skills_text)
            skills_text = re.sub(r"\s+", " ", skills_text).strip(", ")
            return skills_text[:500]  # Cap length for metadata
        return ""

    @staticmethod
    def _extract_experience_years(text: str) -> int:
        """Estimate years of experience from date ranges like 2018-2023."""
        # Pattern: year-year or year–year or year to year
        year_ranges = re.findall(
            r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current|now)",
            text,
            re.IGNORECASE,
        )
        if not year_ranges:
            # Try "X years of experience" pattern
            explicit = re.search(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience", text, re.IGNORECASE)
            if explicit:
                return int(explicit.group(1))
            return 0

        total = 0
        current_year = 2026
        for start_str, end_str in year_ranges:
            start = int(start_str)
            if end_str.lower() in ("present", "current", "now"):
                end = current_year
            else:
                end = int(end_str)
            total += max(0, end - start)
        return min(total, 40)  # Cap at 40

    @staticmethod
    def _extract_education(text: str) -> str:
        """Extract education info."""
        edu_match = re.search(
            r"(?:education|academic|qualifications)\s*[:\-–—]?\s*\n(.*?)(?=\n\s*(?:"
            + _heading_pattern
            + r")\s*[:\-–—]?\s*$|\Z)",
            text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if edu_match:
            edu_text = edu_match.group(1).strip()
            # Take first few lines
            lines = [l.strip() for l in edu_text.split("\n") if l.strip()][:4]
            return " | ".join(lines)[:300]
        return ""


# ---------------------------------------------------------------------------
# Embedding Generator  (Gemini)
# ---------------------------------------------------------------------------
class EmbeddingGenerator:
    """Generate embeddings using Google Gemini gemini-embedding-2-preview."""

    def __init__(self, model: str = EMBEDDING_MODEL, api_key: str | None = None):
        self.model = model
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            # Uses GEMINI_API_KEY or GOOGLE_API_KEY env var
            self.client = genai.Client()

    def embed_documents(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """Embed a list of documents (for storage)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            all_embeddings.extend([e.values for e in result.embeddings])
            if i + batch_size < len(texts):
                time.sleep(0.5)  # Rate-limit courtesy
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (for retrieval)."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values


# ---------------------------------------------------------------------------
# Vector Store  (ChromaDB)
# ---------------------------------------------------------------------------
class VectorStore:
    """ChromaDB-backed vector store for resume chunks."""

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION,
        persist_dir: str = CHROMA_DB_PATH,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ):
        """Add chunks to the collection."""
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> dict:
        """Query the collection by embedding similarity."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )


# ---------------------------------------------------------------------------
# RAG Pipeline  (orchestrator)
# ---------------------------------------------------------------------------
class ResumeRAGPipeline:
    """
    End-to-end pipeline:
      Load PDFs → Chunk → Extract Metadata → Embed → Store in ChromaDB
    """

    def __init__(self, api_key: str | None = None):
        self.loader = PDFLoader()
        self.chunker = ResumeChunker()
        self.extractor = MetadataExtractor()
        self.embedder = EmbeddingGenerator(api_key=api_key)
        self.vector_store = VectorStore()

    def ingest_all(self, resume_dir: str = RESUME_DIR, reset: bool = False) -> dict:
        """
        Ingest all resumes: load, chunk, extract metadata, embed, store.
        Returns stats dict.
        """
        if reset:
            print("Resetting vector store...")
            self.vector_store.reset()

        # 1. Load PDFs
        print("=" * 60)
        print("STEP 1: Loading PDF resumes")
        print("=" * 60)
        documents = self.loader.load_all(resume_dir)

        # 2. Chunk + extract metadata
        print("\n" + "=" * 60)
        print("STEP 2: Chunking & extracting metadata")
        print("=" * 60)
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            # Extract resume-level metadata
            meta = self.extractor.extract(doc["text"])
            # Chunk the resume
            chunks = self.chunker.chunk(doc["text"])
            print(f"  {doc['filename']}: {len(chunks)} chunks, "
                  f"name='{meta['candidate_name']}', "
                  f"exp={meta['experience_years']}yrs")

            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{doc['filename']}_{i}_{chunk['section']}".encode()
                ).hexdigest()

                all_chunks.append(chunk["text"])
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "candidate_name": meta["candidate_name"],
                    "resume_path": f"diverse_resume/{doc['filename']}",
                    "section": chunk["section"],
                    "skills": meta["skills"],
                    "experience_years": meta["experience_years"],
                    "education": meta["education"],
                    "chunk_index": i,
                })

        # 3. Generate embeddings
        print(f"\n{'=' * 60}")
        print(f"STEP 3: Generating embeddings for {len(all_chunks)} chunks")
        print("=" * 60)
        start_time = time.time()
        embeddings = self.embedder.embed_documents(all_chunks)
        embed_time = time.time() - start_time
        print(f"  Embeddings generated in {embed_time:.2f}s")

        # 4. Store in ChromaDB
        print(f"\n{'=' * 60}")
        print("STEP 4: Storing in ChromaDB")
        print("=" * 60)
        # Add in batches to avoid oversized requests
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            self.vector_store.add_documents(
                ids=all_ids[i:end],
                documents=all_chunks[i:end],
                embeddings=embeddings[i:end],
                metadatas=all_metadatas[i:end],
            )
        total = self.vector_store.count()
        print(f"  ✓ Stored {total} chunks in ChromaDB")

        stats = {
            "resumes_loaded": len(documents),
            "total_chunks": len(all_chunks),
            "embedding_time_seconds": round(embed_time, 2),
            "chunks_in_db": total,
        }
        print(f"\n{'=' * 60}")
        print("INGESTION COMPLETE")
        print(json.dumps(stats, indent=2))
        print("=" * 60)
        return stats

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_experience: int | None = None,
    ) -> list[dict]:
        """
        Semantic search over stored resume chunks.
        Optionally filter by minimum experience years.
        """
        query_embedding = self.embedder.embed_query(query)

        where_filter = None
        if min_experience is not None:
            where_filter = {"experience_years": {"$gte": min_experience}}

        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where_filter,
        )

        # Flatten results
        matches = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                matches.append({
                    "candidate_name": meta["candidate_name"],
                    "resume_path": meta["resume_path"],
                    "section": meta["section"],
                    "text": doc,
                    "distance": distance,
                    "similarity": round(1 - distance, 4),
                    "skills": meta["skills"],
                    "experience_years": meta["experience_years"],
                    "education": meta["education"],
                })
        return matches


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume RAG Pipeline")
    sub = parser.add_subparsers(dest="command")

    # Ingest
    ingest_p = sub.add_parser("ingest", help="Ingest all resumes into ChromaDB")
    ingest_p.add_argument("--reset", action="store_true", help="Reset DB before ingesting")
    ingest_p.add_argument("--dir", default=RESUME_DIR, help="Resume directory")

    # Search
    search_p = sub.add_parser("search", help="Search resumes")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--top-k", type=int, default=10)
    search_p.add_argument("--min-exp", type=int, default=None)

    # Info
    sub.add_parser("info", help="Show DB info")

    args = parser.parse_args()
    pipeline = ResumeRAGPipeline()

    if args.command == "ingest":
        pipeline.ingest_all(resume_dir=args.dir, reset=args.reset)
    elif args.command == "search":
        results = pipeline.search(args.query, top_k=args.top_k, min_experience=args.min_exp)
        for r in results:
            print(f"\n{'─' * 40}")
            print(f"  Name: {r['candidate_name']}")
            print(f"  Score: {r['similarity']}")
            print(f"  Section: {r['section']}")
            print(f"  Experience: {r['experience_years']} years")
            print(f"  Excerpt: {r['text'][:200]}...")
    elif args.command == "info":
        print(f"Chunks in DB: {pipeline.vector_store.count()}")
    else:
        parser.print_help()
