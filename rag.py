import re
import os
from dotenv import load_dotenv

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
import streamlit as st


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# =========================================================
# YOUTUBE VIDEO ID EXTRACTION
# =========================================================

def extract_video_id(url):
    """Extract YouTube video ID from common YouTube URL formats."""

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
        r"(?:youtube\.com/embed/)([^?&]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# =========================================================
# CACHED MODELS
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@st.cache_resource(show_spinner=False)
def load_groq():
    return Groq(api_key=GROQ_API_KEY)


# =========================================================
# TRANSCRIPT RETRIEVAL
# =========================================================
# =========================================================
# TRANSCRIPT RETRIEVAL USING FREETRANSCRIPTAPI
# =========================================================

@st.cache_data(show_spinner=False)
def get_transcript(video_id):
    """Fetch YouTube transcript using FreeTranscriptAPI."""

    try:
        youtube_url = (
            f"https://www.youtube.com/watch?v={video_id}"
        )

        api_url = (
            "https://api.freetranscriptapi.com/v1/transcript"
        )

        params = {
            "video_url": youtube_url,
            "lang": "en"
        }

        headers = {}

        # API key is optional
        try:
            api_key = st.secrets.get(
                "FREETRANSCRIPT_API_KEY"
            )
        except Exception:
            api_key = None

        if api_key:
            headers["Authorization"] = (
                f"Bearer {api_key}"
            )

        response = requests.get(
            api_url,
            params=params,
            headers=headers,
            timeout=60
        )

        if response.status_code != 200:
            raise ValueError(
                f"FreeTranscriptAPI error "
                f"{response.status_code}: {response.text}"
            )

        data = response.json()

        transcript_segments = data.get(
            "transcript",
            []
        )

        if not transcript_segments:
            raise ValueError(
                "The API returned an empty transcript."
            )

        text_parts = []

        for segment in transcript_segments:
            if isinstance(segment, dict):
                text = segment.get("text", "")

                if text and text.strip():
                    text_parts.append(text.strip())

        full_text = " ".join(text_parts)

        if not full_text.strip():
            raise ValueError(
                "No readable transcript text was returned."
            )

        return full_text

    except Exception as e:
        st.error(
            f"Transcript error: {type(e).__name__}: {str(e)}"
        )

        return None

# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(text):
    """Create overlapping transcript chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            ", ",
            " ",
            ""
        ]
    )

    raw_chunks = splitter.split_text(text)

    chunks = []

    for index, chunk in enumerate(raw_chunks):
        cleaned = chunk.strip()

        if cleaned:
            chunks.append({
                "id": index,
                "text": cleaned
            })

    return chunks


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):
    return re.findall(
        r"\b[\w'-]+\b",
        text.lower()
    )


# =========================================================
# BUILD VIDEO INDEX
# =========================================================

@st.cache_resource(show_spinner=False)
def build_video_index(video_id):
    """Build FAISS and BM25 indexes."""

    transcript = get_transcript(video_id)
    chunks = create_chunks(transcript)

    if not chunks:
        raise ValueError("No usable transcript chunks found.")

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = load_embeddings()

    vectorstore = FAISS.from_texts(
        chunk_texts,
        embedding=embeddings
    )

    tokenized_chunks = [
        tokenize(text)
        for text in chunk_texts
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return {
        "chunks": chunks,
        "chunk_texts": chunk_texts,
        "vectorstore": vectorstore,
        "bm25": bm25
    }


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(question, vectorstore, k=25):
    """Semantic retrieval using FAISS."""

    return vectorstore.similarity_search(
        question,
        k=k
    )


# =========================================================
# KEYWORD SEARCH
# =========================================================

def keyword_search(question, chunks, bm25, k=25):
    """Keyword retrieval using BM25."""

    question_tokens = tokenize(question)

    scores = bm25.get_scores(question_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices[:k]:
        results.append({
            "id": index,
            "text": chunks[index]["text"]
        })

    return results


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    semantic_documents,
    keyword_chunks,
    k=60
):
    """Combine FAISS and BM25 results."""

    scores = {}
    documents = {}

    # FAISS results
    for rank, document in enumerate(
        semantic_documents,
        start=1
    ):
        text = document.page_content.strip()

        if not text:
            continue

        key = text

        documents[key] = document
        scores[key] = scores.get(key, 0) + (
            1 / (k + rank)
        )

    # BM25 results
    for rank, chunk in enumerate(
        keyword_chunks,
        start=1
    ):
        text = chunk["text"].strip()

        if not text:
            continue

        key = text

        if key not in documents:
            documents[key] = type(
                "Document",
                (),
                {
                    "page_content": text
                }
            )()

        scores[key] = scores.get(key, 0) + (
            1 / (k + rank)
        )

    ranked_documents = sorted(
        documents.values(),
        key=lambda document: scores[
            document.page_content.strip()
        ],
        reverse=True
    )

    return ranked_documents


# =========================================================
# FIND NEIGHBORING CHUNKS
# =========================================================

def add_neighbor_chunks(
    ranked_documents,
    all_chunks,
    neighbor_distance=1
):
    """
    Add nearby transcript chunks.

    This helps with chronological questions where the
    answer may be split across adjacent chunks.
    """

    text_to_index = {
        chunk["text"]: chunk["id"]
        for chunk in all_chunks
    }

    selected_indices = set()

    for document in ranked_documents:
        text = document.page_content.strip()

        if text not in text_to_index:
            continue

        current_index = text_to_index[text]

        start = max(
            0,
            current_index - neighbor_distance
        )

        end = min(
            len(all_chunks),
            current_index + neighbor_distance + 1
        )

        for index in range(start, end):
            selected_indices.add(index)

    expanded_documents = []

    for index in sorted(selected_indices):
        text = all_chunks[index]["text"]

        expanded_documents.append(
            type(
                "Document",
                (),
                {
                    "page_content": text
                }
            )()
        )

    return expanded_documents


# =========================================================
# CROSS-ENCODER RERANKING
# =========================================================

def rerank_documents(
    question,
    documents,
    top_n=10
):
    """Rerank retrieved documents."""

    if not documents:
        return []

    reranker = load_reranker()

    pairs = [
        (
            question,
            document.page_content
        )
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:top_n]


# =========================================================
# COMPLETE RETRIEVAL PIPELINE
# =========================================================

def retrieve_relevant_chunks(
    question,
    video_index
):
    """Hybrid retrieval + neighbor expansion + reranking."""

    chunks = video_index["chunks"]
    vectorstore = video_index["vectorstore"]
    bm25 = video_index["bm25"]

    # 1. Semantic retrieval
    semantic_documents = semantic_search(
        question,
        vectorstore,
        k=25
    )

    # 2. Keyword retrieval
    keyword_chunks = keyword_search(
        question,
        chunks,
        bm25,
        k=25
    )

    # 3. Hybrid fusion
    fused_documents = reciprocal_rank_fusion(
        semantic_documents,
        keyword_chunks
    )

    # 4. Add neighboring transcript chunks
    expanded_documents = add_neighbor_chunks(
        fused_documents[:15],
        chunks,
        neighbor_distance=1
    )

    # 5. Combine original and neighboring documents
    combined_documents = (
        fused_documents[:25] +
        expanded_documents
    )

    # 6. Remove duplicate text
    unique_documents = []
    seen_texts = set()

    for document in combined_documents:
        text = document.page_content.strip()

        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_documents.append(document)

    # 7. Rerank a larger candidate pool
    ranked_documents = rerank_documents(
        question,
        unique_documents,
        top_n=10
    )

    # Debug output
    print("\nQUESTION:", question)
    print("\n===== FINAL RETRIEVED CHUNKS =====")

    for index, (document, score) in enumerate(
        ranked_documents,
        start=1
    ):
        print(
            f"\n--- CHUNK {index} | SCORE: {score:.4f} ---"
        )
        print(document.page_content)

    return ranked_documents


# =========================================================
# GROQ ANSWER GENERATION
# =========================================================

def generate_answer(
    question,
    ranked_documents
):
    """Generate grounded answer using Groq."""

    if not ranked_documents:
        return "I couldn't find a clear answer in the video."

    context_parts = []

    for index, (document, score) in enumerate(
        ranked_documents,
        start=1
    ):
        text = document.page_content.strip()

        if text:
            context_parts.append(
                f"[Evidence {index}]\n{text}"
            )

    context = "\n\n---\n\n".join(context_parts)

    # Prevent excessively large prompts
    context = context[:18000]

    prompt = f"""
You are a precise question-answering assistant for YouTube videos.

Answer the user's question using ONLY the supplied evidence.

USER QUESTION:
{question}

SUPPLIED EVIDENCE:
{context}

INSTRUCTIONS:

1. Answer the exact question directly.

2. Do not use outside knowledge.

3. Do not guess or invent missing information.

4. For questions containing words such as:
   - after
   - before
   - next
   - previous
   - first
   - second
   - later
   - earlier
   - following
   - preceding

   carefully reconstruct the sequence from the evidence.



7. If multiple evidence passages describe the same event,
   combine them.

8. If the evidence contains a transcription error or
   inconsistent spelling, use the surrounding evidence
   to understand the intended entity, but do not invent
   unsupported facts.

9. Give the direct answer first.

10. Normally answer in 1-3 concise sentences.

11. Do not mention:
    - transcript
    - evidence
    - chunks
    - FAISS
    - BM25
    - embeddings
    - reranking
    - RAG
    - context
    - retrieval

12. Do not repeat the user's question.

13. If the supplied information genuinely does not support
    an answer, say:
    I couldn't find a clear answer in the video.
"""

    client = load_groq()

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            top_p=1,
            seed=42,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=500
        )

        answer = response.choices[0].message.content

        if not answer:
            return "I couldn't find a clear answer in the video."

        answer = answer.strip()

        answer = re.sub(
            r"^\s*(FINAL ANSWER|ANSWER)\s*:\s*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"\[?\s*TRANSCRIPT SECTION\s*\d+\s*\]?",
            "",
            answer,
            flags=re.IGNORECASE
        )

        return answer.strip()

    except Exception as e:
        raise Exception(
            f"Groq error: {str(e)}"
        )