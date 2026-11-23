"""
Retrieval-Augmented Generation (RAG) Engine.
Retrieves relevant medical context using vector similarity and queries an LLM
with strict clinical grounding instructions and source citation requirements.
Includes an intelligent deterministic fallback mode when API keys are not provided.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None

# Optional LLM SDKs
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None


SYSTEM_PROMPT = """You are a specialized clinical AI assistant providing information strictly grounded in verified healthcare guidelines.

STRICT INSTRUCTIONS:
1. Answer ONLY using the facts directly stated in the provided context passages below.
2. If the context does not contain the answer, or if the question asks about something outside the provided healthcare documents, you MUST explicitly state:
   "I cannot answer this based on the provided healthcare documents."
3. Never extrapolate, speculate, make assumptions, or introduce external medical claims not present in the excerpts.
4. At the end of your answer, explicitly list each source document name used under a "Sources:" heading.
5. Maintain a professional, objective, and empathetic clinical tone. Always include a short reminder that this is for informational purposes and not a diagnostic substitute.
"""


class RAGEngine:
    def __init__(
        self,
        vector_store_dir: Path = settings.VECTOR_STORE_DIR,
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
    ):
        self.vector_store_dir = vector_store_dir
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.embedder = None
        self.index = None
        self.metadata = None
        self.embeddings = None
        self._load_store()

    def _load_store(self):
        """Load vector store, embeddings, and metadata."""
        meta_path = self.vector_store_dir / "metadata.json"
        index_path = self.vector_store_dir / "index.faiss"
        npy_path = self.vector_store_dir / "embeddings.npy"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Vector store metadata not found at {meta_path}. Please run `python src/rag_indexer.py` first."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if SentenceTransformer is not None:
            self.embedder = SentenceTransformer(self.embedding_model_name)

        if faiss is not None and index_path.exists():
            self.index = faiss.read_index(str(index_path))
        elif npy_path.exists():
            self.embeddings = np.load(npy_path)

    def retrieve(self, query: str, top_k: int = settings.TOP_K) -> List[Dict[str, Any]]:
        """
        Embed user query and retrieve top-k chunks with similarity scores.
        Returns list of dicts with text, source_name, chunk_id, and similarity_score.
        """
        if self.embedder is None:
            raise RuntimeError("SentenceTransformer is not initialized.")

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        chunks = self.metadata["chunks"]
        retrieved = []

        if self.index is not None:
            # FAISS Inner Product search on normalized vectors = Cosine Similarity
            scores, indices = self.index.search(query_embedding, top_k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunks):
                    chunk_data = dict(chunks[idx])
                    chunk_data["similarity_score"] = float(score)
                    retrieved.append(chunk_data)
        else:
            # Fallback pure-numpy cosine similarity
            if self.embeddings is None:
                npy_path = self.vector_store_dir / "embeddings.npy"
                self.embeddings = np.load(npy_path)
            sims = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            for idx in top_indices:
                chunk_data = dict(chunks[idx])
                chunk_data["similarity_score"] = float(sims[idx])
                retrieved.append(chunk_data)

        return retrieved

    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Query Gemini using official SDK if API key is present."""
        api_key = settings.GEMINI_API_KEY
        if not api_key or api_key == "your_gemini_api_key_here":
            return None

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=settings.TEMPERATURE,
                ),
            )
            return response.text
        except Exception as e:
            print(f"Gemini API query error: {e}")
            return None

    def _query_openai(self, prompt: str) -> Optional[str]:
        """Query OpenAI if API key is present."""
        api_key = settings.OPENAI_API_KEY
        if not api_key or api_key == "your_openai_api_key_here" or openai is None:
            return None

        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API query error: {e}")
            return None

    def _deterministic_grounded_answer(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Deterministic, offline extraction engine.
        Synthesizes an answer strictly from top retrieved excerpts and appends citations.
        Used when no live LLM API keys are provided.
        """
        # Collect unique sources
        sources = sorted(list(set(c["source_name"] for c in context_chunks)))
        
        # Build answer points by extracting the most salient sentences
        bullet_points = []
        for c in context_chunks:
            lines = [l.strip() for l in c["text"].split("\n") if l.strip() and not l.startswith("#")]
            for line in lines:
                if len(line) > 30 and not line.endswith(":"):
                    clean_line = line.lstrip("-*1234567890. ")
                    if clean_line and clean_line not in bullet_points and len(bullet_points) < 6:
                        bullet_points.append(clean_line)

        answer_body = "\n".join([f"• {pt}" for pt in bullet_points])
        sources_list = "\n".join([f"- `{src}`" for src in sources])

        response = (
            f"Based on the provided clinical documentation:\n\n"
            f"{answer_body}\n\n"
            f"**Sources:**\n"
            f"{sources_list}\n\n"
            f"*Disclaimer: This summary is generated from verified medical guidelines for informational screening purposes and does not substitute for a clinical consultation with a licensed physician.*"
        )
        return response

    def ask(self, query: str, top_k: int = settings.TOP_K) -> Dict[str, Any]:
        """
        Complete RAG pipeline:
        1. Retrieve top-k chunks
        2. Evaluate similarity threshold for hallucination refusal
        3. Query LLM (Gemini, OpenAI, or deterministic fallback)
        4. Return response, sources, and chunks
        """
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        # Check maximum similarity score
        max_score = retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.0

        # Strict Out-of-Domain refusal guardrail
        if not retrieved_chunks or max_score < self.similarity_threshold:
            return {
                "query": query,
                "answer": "I cannot answer this based on the provided healthcare documents.",
                "sources": [],
                "retrieved_chunks": retrieved_chunks,
                "max_similarity": max_score,
                "grounded": False,
                "refusal": True,
            }

        # Filter chunks that meet the relevance cutoff
        relevant_chunks = [c for c in retrieved_chunks if c["similarity_score"] >= self.similarity_threshold]
        if not relevant_chunks:
            relevant_chunks = retrieved_chunks[:2]

        # Format context prompt
        context_str = "\n\n---\n\n".join(
            [f"[Source: {c['source_name']}]\n{c['text']}" for c in relevant_chunks]
        )
        user_prompt = (
            f"CONTEXT PASSAGES:\n{context_str}\n\n"
            f"PATIENT / USER QUESTION: {query}\n\n"
            f"Provide an answer strictly following the grounding instructions."
        )

        answer = None
        # Try configured LLM provider
        if settings.LLM_PROVIDER.lower() == "gemini":
            answer = self._query_gemini(user_prompt)
        elif settings.LLM_PROVIDER.lower() == "openai":
            answer = self._query_openai(user_prompt)

        # Fallback to alternative provider or deterministic engine
        if answer is None:
            if settings.LLM_PROVIDER.lower() != "gemini":
                answer = self._query_gemini(user_prompt)
            if answer is None and settings.LLM_PROVIDER.lower() != "openai":
                answer = self._query_openai(user_prompt)

        if answer is None:
            # Deterministic offline engine
            answer = self._deterministic_grounded_answer(query, relevant_chunks)

        sources = sorted(list(set(c["source_name"] for c in relevant_chunks)))

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": relevant_chunks,
            "max_similarity": round(max_score, 4),
            "grounded": True,
            "refusal": False,
        }

    def explain_patient_risk(self, risk_result: Dict[str, Any]) -> str:
        """
        Generate a personalized clinical explanation for an ML risk prediction.
        """
        prob = risk_result["risk_percentage"]
        tier = risk_result["risk_tier"]
        flags = risk_result.get("clinical_flags", [])
        p_data = risk_result.get("patient_data", {})

        query = (
            f"A patient has been assessed with a {tier} of diabetes ({prob}% calculated risk). "
            f"Key clinical indicators include: Glucose {p_data.get('Glucose', 'N/A')} mg/dL, "
            f"Blood Pressure {p_data.get('BloodPressure', 'N/A')} mmHg, "
            f"BMI {p_data.get('BMI', 'N/A')}, Age {p_data.get('Age', 'N/A')}. "
            f"Clinical flags noted: {'; '.join(flags) if flags else 'None'}. "
            f"Explain what these risk factors mean according to clinical guidelines, "
            f"recommend relevant diagnostic tests (like HbA1c), and discuss lifestyle modifications."
        )

        rag_output = self.ask(query, top_k=4)
        return rag_output["answer"]


if __name__ == "__main__":
    print("Testing RAG Engine initialization and retrieval...")
    engine = RAGEngine()
    
    # Test In-domain
    test_q1 = "What are the early symptoms of diabetes?"
    res1 = engine.ask(test_q1)
    print(f"\nQ: {test_q1}")
    print(f"Refusal: {res1['refusal']}")
    print(f"Max Sim: {res1['max_similarity']}")
    print(f"Sources: {res1['sources']}")
    print(f"Answer snippet: {res1['answer'][:200]}...")

    # Test Out-of-domain
    test_q2 = "What is the cure for Alzheimer's disease?"
    res2 = engine.ask(test_q2)
    print(f"\nQ: {test_q2}")
    print(f"Refusal: {res2['refusal']}")
    print(f"Max Sim: {res2['max_similarity']}")
    print(f"Answer: {res2['answer']}")
