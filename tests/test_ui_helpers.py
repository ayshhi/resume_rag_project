import unittest

from resume_assistant.rag_pipeline import generate_interview_questions


class UIHelpersTests(unittest.TestCase):
    def test_question_generator_is_richer(self) -> None:
        context = "Built APIs with FastAPI, used Docker for deployment, and led a team of engineers."
        questions = generate_interview_questions(context)
        self.assertGreaterEqual(len(questions), 4)
        self.assertTrue(any("docker" in question.lower() for question in questions))
        self.assertTrue(any("team" in question.lower() for question in questions))


if __name__ == "__main__":
    unittest.main()
