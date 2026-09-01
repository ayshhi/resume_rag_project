from __future__ import annotations

import argparse
from pathlib import Path

from resume_assistant.app import answer_question, build_context_index, chunk_text, generate_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume RAG interview assistant")
    parser.add_argument("resume_path", help="Path to a resume text file")
    parser.add_argument("question", nargs="?", default="What experience is highlighted in the resume?", help="Question to ask")
    args = parser.parse_args()

    text = Path(args.resume_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    index = build_context_index(chunks)
    print("Answer:")
    print(answer_question(args.question, index))
    print("\nSuggested interview questions:")
    for question in generate_questions(index):
        print(f"- {question}")


if __name__ == "__main__":
    main()
