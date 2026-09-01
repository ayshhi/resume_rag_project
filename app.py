from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from resume_assistant.rag_pipeline import (
    answer_with_openrouter,
    build_faiss_index,
    classify_openrouter_key,
    extract_text_from_pdf,
    generate_interview_questions,
    identify_strengths_and_gaps,
    retrieve_relevant_chunks,
    split_text_into_chunks,
    suggest_resume_improvements,
    summarize_candidate_profile,
)

load_dotenv()


def _get_effective_openrouter_key() -> str | None:
    raw_key = os.getenv("OPENROUTER_API_KEY")
    if raw_key is None:
        return None
    key = str(raw_key).strip().strip('"').strip("'")
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
        return None
    if key.lower().startswith("your_") or ("your" in key.lower() and "here" in key.lower()):
        return None
    return key


st.set_page_config(page_title="Resume Interview Assistant", page_icon="🧠", layout="wide")

st.markdown(
    """
    <div style="background: linear-gradient(90deg, #4f46e5, #0ea5e9); padding: 1.4rem; border-radius: 12px; color: white;">
        <h1 style="margin:0; font-size:2rem;">Resume Interview Assistant</h1>
        <p style="margin:0.25rem 0 0 0; font-size:1rem;">Upload a resume, inspect the most relevant passages, and ask grounded interview questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

openrouter_key = _get_effective_openrouter_key()

with st.expander("OpenRouter setup (step by step)", expanded=not bool(openrouter_key)):
    st.markdown("1. Open [OpenRouter](https://openrouter.ai/keys) and create an API key.")
    st.write("2. Paste the key below or add it to the .env file as OPENROUTER_API_KEY=real_key_here")
    st.write("3. Restart the app so Streamlit loads the new environment variables.")
    st.write("4. Reload the page and the app will use OpenRouter for grounded answers.")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Get API key", use_container_width=True):
            st.link_button("Open OpenRouter", "https://openrouter.ai/keys")
    with col_b:
        api_key_input = st.text_input("Optional: paste your OpenRouter API key here for this session", type="password")
    if api_key_input:
        cleaned_key = api_key_input.strip().strip('"').strip("'")
        ok, message = classify_openrouter_key(cleaned_key)
        if not ok:
            st.warning(message)
        else:
            os.environ["OPENROUTER_API_KEY"] = cleaned_key
            if "temporarily" in message.lower() or "quota" in message.lower() or "rate" in message.lower():
                st.info(message)
            else:
                st.success(message)
            openrouter_key = _get_effective_openrouter_key()
    if st.button("Test key", use_container_width=True):
        if not api_key_input:
            st.warning("Paste a key first.")
        else:
            ok, message = classify_openrouter_key(api_key_input)
            if ok:
                if "temporarily" in message.lower() or "quota" in message.lower() or "rate" in message.lower():
                    st.info(message)
                else:
                    st.success(message)
                os.environ["OPENROUTER_API_KEY"] = api_key_input.strip().strip('"').strip("'")
                openrouter_key = _get_effective_openrouter_key()
            else:
                st.warning(message)

if not openrouter_key:
    st.caption("OpenRouter API key is not configured. The app will use a local fallback path for common resume questions.")
else:
    st.caption("OpenRouter API key detected. Answers will be generated with OpenRouter using the retrieved resume context.")

uploaded_file = st.file_uploader("Choose a resume PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        st.error("No text could be extracted from the PDF.")
        st.stop()

    chunks = split_text_into_chunks(text)
    if not chunks:
        st.error("The resume text was empty after chunking.")
        st.stop()

    if "rag_store" not in st.session_state:
        st.session_state.rag_store = build_faiss_index(chunks, output_dir="./data")

    st.success(f"Indexed {len(chunks)} chunks from the uploaded resume.")

    question = st.text_input("Ask a question about the resume", "What experience does the candidate highlight?")
    if st.button("Get answer", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question before asking.")
        else:
            relevant_chunks = retrieve_relevant_chunks(question, st.session_state.rag_store)
            context = "\n\n".join(relevant_chunks)
            api_key = _get_effective_openrouter_key()
            answer = answer_with_openrouter(question, context, api_key=api_key)

            st.subheader("Answer")
            st.info(answer)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Retrieved context")
                st.text_area("Context", context, height=220)
            with col2:
                st.subheader("Candidate summary")
                st.write(summarize_candidate_profile(context, api_key=api_key))
                st.subheader("Strengths & gaps")
                st.write(identify_strengths_and_gaps(context, api_key=api_key))

            st.subheader("Interview questions")
            for item in generate_interview_questions(context, api_key=api_key):
                st.write(f"- {item}")

            st.subheader("Resume improvement suggestions")
            for item in suggest_resume_improvements(context, api_key=api_key):
                st.write(f"- {item}")
else:
    st.info("Upload a PDF resume to begin.")