import unittest
from pathlib import Path

from resume_assistant.app import (
    answer_question,
    build_context_index,
    chunk_text,
    generate_questions,
)
from resume_assistant.rag_pipeline import (
    answer_with_gemini,
    summarize_candidate_profile,
    identify_strengths_and_gaps,
)


class AssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resume_text = (
            "Jane Doe\n"
            "Senior Python Engineer\n"
            "Built APIs with FastAPI, used pandas for data cleaning, and deployed services with Docker.\n"
            "Led a team of five engineers and delivered analytics features for enterprise clients."
        )

    def test_chunking_and_indexing(self) -> None:
        chunks = chunk_text(self.resume_text, chunk_size=40, overlap=10)
        self.assertGreater(len(chunks), 1)
        index = build_context_index(chunks)
        self.assertEqual(len(index["chunks"]), len(chunks))
        self.assertTrue(Path(index["db_path"]).exists())
        self.assertGreaterEqual(index["vector_count"], len(chunks))

    def test_answer_is_grounded(self) -> None:
        chunks = chunk_text(self.resume_text, chunk_size=40, overlap=10)
        index = build_context_index(chunks)
        answer = answer_question("What Python frameworks has Jane used?", index)
        self.assertIn("fastapi", answer.lower())

    def test_generate_questions(self) -> None:
        chunks = chunk_text(self.resume_text, chunk_size=40, overlap=10)
        index = build_context_index(chunks)
        questions = generate_questions(index)
        self.assertTrue(questions)
        self.assertTrue(any("python" in question.lower() for question in questions))

    def test_fallback_answer_for_cgpa(self) -> None:
        context = "Education: B.Tech in Computer Science, CGPA 8.7/10"
        answer = answer_with_gemini("What is the CGPA?", context, api_key=None)
        self.assertIn("8.7", answer)
        self.assertIn("cgpa", answer.lower())

    def test_fallback_answer_for_skills(self) -> None:
        context = "Skills: Python, FastAPI, Docker, pandas"
        answer = answer_with_gemini("What skills are mentioned?", context, api_key=None)
        self.assertIn("python", answer.lower())
        self.assertIn("fastapi", answer.lower())

    def test_summary_and_strengths(self) -> None:
        context = "Built APIs with FastAPI and Docker. Led a team of engineers."
        summary = summarize_candidate_profile(context)
        strengths = identify_strengths_and_gaps(context)
        self.assertIn("fastapi", summary.lower())
        self.assertIn("strengths", strengths.lower())


if __name__ == "__main__":
    unittest.main()
