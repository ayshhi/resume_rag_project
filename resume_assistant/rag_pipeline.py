from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import re

import faiss
import numpy as np
import pdfplumber
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_openrouter_model_candidates() -> List[str]:
    """Return candidate OpenRouter model names in a pragmatic order."""
    env_model = os.getenv("OPENROUTER_MODEL", "").strip()
    candidates: List[str] = []
    if env_model:
        candidates.append(env_model)
    candidates.extend([
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-haiku",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.1-8b-instruct",
    ])
    seen: set[str] = set()
    ordered: List[str] = []
    for name in candidates:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _call_openrouter(prompt: str, api_key: str) -> tuple[str, str]:
    """Call the OpenRouter chat completions endpoint, trying candidate models in order."""
    last_error: Exception | None = None
    model_candidates = _get_openrouter_model_candidates()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for attribution/rate-limit purposes.
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "resume-rag-assistant"),
    }

    for model_name in model_candidates:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code >= 400:
                error_text = response.text.lower()
                last_error = RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
                if response.status_code == 404 or "not found" in error_text or "unsupported" in error_text:
                    continue
                if response.status_code in (401, 403):
                    raise last_error
                raise last_error

            data = response.json()
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "") or ""
                if isinstance(text, str) and text.strip():
                    return text.strip(), model_name
        except requests.RequestException as exc:  # pragma: no cover
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("No OpenRouter model responded successfully")


def classify_openrouter_key(api_key: str | None) -> tuple[bool, str]:
    """Validate that an OpenRouter API key is usable by attempting a tiny request."""
    if not api_key:
        return False, "No key provided."
    key = str(api_key).strip().strip('"').strip("'")
    placeholder_values = {
        "",
        "your_openrouter_api_key_here",
        "your_key_here",
        "your_api_key_here",
        "api_key_here",
        "changeme",
        "placeholder",
    }
    if key.lower() in placeholder_values:
        return False, "That looks like a placeholder value."
    if key.lower().startswith("your_") or ("your" in key.lower() and "here" in key.lower()):
        return False, "That looks like a placeholder value."
    if not key.lower().startswith("sk-or-"):
        # OpenRouter keys conventionally start with "sk-or-"; warn but still attempt validation.
        pass

    try:
        text, model_name = _call_openrouter("Reply with the single word OK.", key)
        if text.strip():
            return True, f"Key validated successfully through {model_name}."
        return True, "Key format looks valid, but OpenRouter returned no confirmation text."
    except Exception as exc:  # pragma: no cover
        error_text = str(exc).lower()
        if "quota" in error_text or "rate limit" in error_text or "retry" in error_text or "temporarily" in error_text or "429" in error_text:
            return True, f"The key looks valid, but OpenRouter is temporarily rate-limiting or quota-blocking requests: {exc}"
        if "api key" in error_text or "invalid" in error_text or "forbidden" in error_text or "permission" in error_text or "401" in error_text or "403" in error_text:
            return False, f"OpenRouter rejected the key: {exc}"
        return False, f"Validation failed: {exc}"


def _repair_glued_words(text: str) -> str:
    """Insert spaces where PDF text extraction merged adjacent words with no whitespace.

    Some PDFs (especially resume templates with tables/text-boxes) never encode an
    actual space glyph between visually-adjacent words, so both pdfplumber and pypdf
    can emit strings like "StJamesHospitalChalakudy". This heuristically re-splits
    such runs on case and letter/digit boundaries without touching real acronyms.
    """
    # Split lowercase->Uppercase boundaries: "JamesHospital" -> "James Hospital"
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Split letter->digit and digit->letter boundaries: "Ph2002" -> "Ph 2002", "2002gmail" -> "2002 gmail"
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    # Collapse any resulting multiple spaces.
    return re.sub(r"[ \t]+", " ", text)


def _extract_text_pdfplumber(source: Any) -> str:
    pages_text: List[str] = []
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            # x_tolerance/y_tolerance control how close characters must be to be
            # merged into the same word; smaller values preserve real word gaps
            # that pypdf's default extraction tends to swallow.
            page_text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            if page_text:
                pages_text.append(page_text)
    return "\n".join(pages_text).strip()


def _extract_text_pypdf(source: Any) -> str:
    reader = PdfReader(source)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page for page in pages if page).strip()


def extract_text_from_pdf(uploaded_file: Any) -> str:
    """Extract text from a Streamlit UploadedFile or file path.

    Tries pdfplumber first (better at preserving word boundaries via character
    positions), and falls back to pypdf if pdfplumber fails or returns nothing.
    Either way, a word-spacing repair pass runs afterward as a safety net for
    PDFs that never encode real spaces between visually-adjacent words.
    """
    if hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
    else:
        raw_bytes = Path(str(uploaded_file)).read_bytes()

    text = ""
    try:
        text = _extract_text_pdfplumber(BytesIO(raw_bytes))
    except Exception:  # pragma: no cover
        text = ""

    if not text.strip():
        try:
            text = _extract_text_pypdf(BytesIO(raw_bytes))
        except Exception:  # pragma: no cover
            text = ""

    if not text.strip():
        return ""

    # Only run the glued-word repair on lines that look suspiciously dense
    # (very long tokens with no spaces), so normal well-formed PDFs are untouched.
    repaired_lines = []
    for line in text.split("\n"):
        words = line.split(" ")
        looks_glued = any(len(word) > 25 for word in words)
        repaired_lines.append(_repair_glued_words(line) if looks_glued else line)
    return "\n".join(repaired_lines).strip()


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into smaller chunks for retrieval."""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = max(chunk_size - overlap, 1)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
    return chunks


def build_faiss_index(chunks: List[str], output_dir: str | None = None) -> Dict[str, Any]:
    """Create embeddings and a local FAISS index for semantic search."""
    if output_dir is None:
        output_dir = str(Path("./data"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(output_path / "faiss_index.index"))
    (output_path / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    return {
        "chunks": chunks,
        "index_path": str(output_path / "faiss_index.index"),
        "chunks_path": str(output_path / "chunks.json"),
        "model": model,
    }


def retrieve_relevant_chunks(query: str, store: Dict[str, Any], top_k: int = 4) -> List[str]:
    """Retrieve the most relevant chunks using the FAISS index."""
    model = store["model"]
    index = faiss.read_index(store["index_path"])
    chunks = json.loads(Path(store["chunks_path"]).read_text(encoding="utf-8"))
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(embedding, min(top_k, len(chunks)))
    ranked_chunks: List[str] = []
    for idx in indices[0]:
        if idx < len(chunks):
            ranked_chunks.append(chunks[int(idx)])
    return ranked_chunks


def _extract_resume_value(question: str, context: str) -> str | None:
    """Extract a direct, grounded answer from the resume context when no OpenRouter key is available."""
    import re

    q = question.lower()
    context_lower = context.lower()
    if "cgpa" in q:
        match = re.search(r"cgpa\s*[:#-]?\s*(\d+(?:\.\d+)?)", context_lower)
        if match:
            return f"The CGPA is {match.group(1)}."
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", context)
        if match:
            return f"The CGPA is {match.group(1)}/10."
        return None

    if "graduation" in q or "graduate" in q or "year" in q and "graduat" in q:
        match = re.search(r"(20\d{2}|19\d{2})", context)
        if match:
            return f"The graduation year mentioned is {match.group(1)}."
        return None

    if "skill" in q or "skills" in q:
        skills = []
        for token in ["python", "fastapi", "docker", "pandas", "sql", "java", "c++", "c#", "javascript", "react"]:
            if token in context_lower:
                skills.append(token)
        if skills:
            return "The resume mentions skills such as: " + ", ".join(skills) + "."
        return None

    if "project" in q or "projects" in q:
        if "fastapi" in context_lower or "api" in context_lower:
            return "The resume mentions projects involving API development and deployment-related work."
        return None

    if "experience" in q or "worked" in q or "work" in q:
        if "team" in context_lower or "lead" in context_lower or "engineer" in context_lower:
            return "The resume highlights engineering and leadership experience."
        return None

    return None


def _call_openrouter_json(prompt: str, api_key: str) -> Any:
    """Call OpenRouter expecting a raw JSON response body, and parse it."""
    import re

    raw_text, _ = _call_openrouter(prompt, api_key)
    cleaned = raw_text.strip()
    # Strip ```json ... ``` fences if the model wrapped its output despite instructions.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return json.loads(cleaned)


def answer_with_openrouter(question: str, context: str, api_key: str | None = None) -> str:
    """Generate a grounded answer from the retrieved resume context using OpenRouter if available."""
    prompt = f"""You are a strict resume interview assistant for recruiters. Use only the resume context below to answer the user's question.

Resume context:
{context}

User question:
{question}

Instructions:
1. Answer only from the resume context.
2. If the question is about interview readiness, skills, projects, education, experience, or strengths, provide a concise and evidence-based answer grounded in the resume.
3. If the answer is missing from the resume context, explicitly say that the resume does not provide enough evidence.
4. Do not infer or invent facts.
5. Do not add background knowledge or general advice.
6. Keep the response short, factual, and recruiter-friendly.
7. If the question asks for a specific value such as CGPA, graduation year, skills, or projects, return that value directly.
8. If the user asks for interview questions, return a short list of relevant interview prompts based only on the resume content.
"""
    if not api_key:
        extracted = _extract_resume_value(question, context)
        if extracted:
            return f"Based on the resume context, {extracted}"
        return "OpenRouter API key not provided. Using the resume context fallback.\n\n" + context[:800]

    try:
        response_text, _ = _call_openrouter(prompt, api_key)
        return response_text.strip()
    except Exception as exc:  # pragma: no cover
        error_detail = str(exc)
        lowered = error_detail.lower()
        if "api key" in lowered or "invalid" in lowered or "401" in lowered or "403" in lowered:
            extracted = _extract_resume_value(question, context)
            if extracted:
                return f"OpenRouter key was rejected ({error_detail}). Falling back to the resume context.\n\nBased on the resume context, {extracted}"
            return f"OpenRouter key was rejected ({error_detail}). Falling back to the resume context.\n\n{context[:800]}"
        return f"OpenRouter call failed: {error_detail}\n\nContext used:\n{context[:800]}"


def _generate_interview_questions_fallback(context: str) -> List[str]:
    """Keyword-based fallback used when no OpenRouter key is available or the call fails."""
    lower = context.lower()
    questions: List[str] = []
    if "python" in lower or "fastapi" in lower:
        questions.append("How did you apply Python and FastAPI in the projects you described?")
        questions.append("Which Python-based problem was most challenging for you, and how did you solve it?")
    if "docker" in lower:
        questions.append("Can you walk through a deployment or containerization experience involving Docker?")
        questions.append("How did your deployment work improve reliability or delivery speed?")
    if "team" in lower or "lead" in lower:
        questions.append("How did you lead or coordinate a team to deliver a successful outcome?")
        questions.append("What was the impact of your leadership on the project or business?")
    if "api" in lower or "fastapi" in lower:
        questions.append("What design trade-offs did you consider when building the APIs mentioned in your experience?")
    if "analytics" in lower or "data" in lower or "pandas" in lower:
        questions.append("How did you use data processing or analytics tools to solve a concrete business problem?")
    if not questions:
        questions.append("What achievement from the resume would you like to discuss in more depth?")
    return questions[:6]


def generate_interview_questions(context: str, api_key: str | None = None) -> List[str]:
    """Generate recruiter-style interview questions grounded in the actual resume context."""
    if not api_key:
        return _generate_interview_questions_fallback(context)

    prompt = f"""You are a recruiter preparing interview questions for a candidate, based only on their resume.

Resume context:
{context}

Write 5-6 specific interview questions that reference the candidate's actual projects, roles, tools, or achievements named in the resume context above. Do not ask about skills, tools, or technologies that are not mentioned in the context. Do not invent details.

Respond with ONLY a raw JSON array of strings, no markdown fences, no preamble. Example format:
["question one", "question two"]
"""
    try:
        parsed = _call_openrouter_json(prompt, api_key)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed) and parsed:
            return parsed[:6]
    except Exception:  # pragma: no cover
        pass
    return _generate_interview_questions_fallback(context)


def _summarize_candidate_profile_fallback(context: str) -> str:
    """Keyword-based fallback used when no OpenRouter key is available or the call fails."""
    if not context:
        return "No candidate details available."
    lower = context.lower()
    summary_parts = []
    if "python" in lower:
        summary_parts.append("strong Python-based software development experience")
    if "docker" in lower:
        summary_parts.append("deployment and containerization experience")
    if "fastapi" in lower:
        summary_parts.append("FastAPI experience")
    if "team" in lower or "lead" in lower:
        summary_parts.append("leadership or team collaboration experience")
    if not summary_parts:
        return "Candidate highlights: " + context[:800]
    return "Candidate profile: " + "; ".join(summary_parts) + "."


def summarize_candidate_profile(context: str, api_key: str | None = None) -> str:
    """Create a candidate summary grounded in the actual resume context."""
    if not context:
        return "No candidate details available."
    if not api_key:
        return _summarize_candidate_profile_fallback(context)

    prompt = f"""Summarize this candidate's background in 2-3 sentences, based only on the resume context below. Mention their actual field, degree, key projects, and domain (e.g. electronics, ML, web development, finance) — whatever is truly reflected in the text. Do not assume a software engineering background unless the resume shows one. Do not invent facts.

Resume context:
{context}

Respond with ONLY the summary text, no markdown, no preamble, no quotes.
"""
    try:
        text, _ = _call_openrouter(prompt, api_key)
        if text.strip():
            return text.strip()
    except Exception:  # pragma: no cover
        pass
    return _summarize_candidate_profile_fallback(context)


def _identify_strengths_and_gaps_fallback(context: str) -> str:
    """Keyword-based fallback used when no OpenRouter key is available or the call fails."""
    lower = context.lower()
    strengths = []
    gaps = []
    if "python" in lower:
        strengths.append("Python experience")
    if "docker" in lower:
        strengths.append("Deployment/containerization")
    if "team" in lower or "lead" in lower:
        strengths.append("Leadership or team coordination")
    if "fastapi" in lower:
        strengths.append("API development")
    if not strengths:
        return "No strong signals found in the provided resume context."
    if "metrics" not in lower and "impact" not in lower and "deliver" not in lower:
        gaps.append("Quantified achievements and impact could be strengthened")
    return "Strengths: " + ", ".join(strengths) + "\nGaps: " + (", ".join(gaps) if gaps else "No obvious gaps detected")


def identify_strengths_and_gaps(context: str, api_key: str | None = None) -> str:
    """Identify strengths and gaps grounded in the actual resume context."""
    if not api_key:
        return _identify_strengths_and_gaps_fallback(context)

    prompt = f"""Analyze this resume context and identify the candidate's real strengths and any gaps, based only on what's actually written. Reference their actual domain (whatever it is) and specific projects/tools mentioned.

Resume context:
{context}

Respond with ONLY a raw JSON object, no markdown fences, no preamble, in this exact shape:
{{"strengths": ["strength one", "strength two"], "gaps": ["gap one"]}}
If there are no clear gaps, use an empty list for "gaps".
"""
    try:
        parsed = _call_openrouter_json(prompt, api_key)
        if isinstance(parsed, dict):
            strengths = parsed.get("strengths") or []
            gaps = parsed.get("gaps") or []
            if strengths:
                strengths_line = "Strengths: " + ", ".join(str(s) for s in strengths)
                gaps_line = "Gaps: " + (", ".join(str(g) for g in gaps) if gaps else "No obvious gaps detected")
                return strengths_line + "\n" + gaps_line
    except Exception:  # pragma: no cover
        pass
    return _identify_strengths_and_gaps_fallback(context)


def _suggest_resume_improvements_fallback(context: str) -> List[str]:
    """Keyword-based fallback used when no OpenRouter key is available or the call fails."""
    suggestions: List[str] = []
    if "python" in context.lower():
        suggestions.append("Add a short bullet listing measurable Python project outcomes.")
    if "docker" in context.lower():
        suggestions.append("Mention containerization wins, deployment scale, and reliability improvements.")
    if "team" in context.lower() or "lead" in context.lower():
        suggestions.append("Highlight leadership outcomes such as team size, delivery timelines, and business impact.")
    if not suggestions:
        suggestions.append("Include quantified achievements, metrics, and tools to make the resume more compelling.")
    return suggestions


def suggest_resume_improvements(context: str, api_key: str | None = None) -> List[str]:
    """Suggest practical resume improvements grounded in the actual resume context."""
    if not api_key:
        return _suggest_resume_improvements_fallback(context)

    prompt = f"""Suggest 3-5 concrete, actionable improvements for this resume, based only on what's actually written below. Reference the candidate's real projects, field, and achievements. Do not suggest generic advice unrelated to their actual background.

Resume context:
{context}

Respond with ONLY a raw JSON array of strings, no markdown fences, no preamble. Example format:
["suggestion one", "suggestion two"]
"""
    try:
        parsed = _call_openrouter_json(prompt, api_key)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed) and parsed:
            return parsed[:5]
    except Exception:  # pragma: no cover
        pass
    return _suggest_resume_improvements_fallback(context)