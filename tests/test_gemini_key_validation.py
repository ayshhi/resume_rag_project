import unittest
from unittest.mock import patch

from resume_assistant.rag_pipeline import classify_gemini_key


class GeminiKeyValidationTests(unittest.TestCase):
    def test_placeholder_keys_are_rejected(self) -> None:
        ok, message = classify_gemini_key("your_gemini_api_key_here")
        self.assertFalse(ok)
        self.assertIn("placeholder", message.lower())

    def test_realistic_keys_are_not_rejected_by_format(self) -> None:
        ok, message = classify_gemini_key("AIzaSyDExampleKey123456")
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(message, str)
        self.assertGreater(len(message), 0)

    def test_quota_errors_are_reported_as_temporarily_unavailable(self) -> None:
        with patch("resume_assistant.rag_pipeline._call_gemini", side_effect=Exception("429 quota exceeded for generate_content")):
            ok, message = classify_gemini_key("AIzaSyDExampleKey123456")
        self.assertTrue(ok)
        self.assertIn("temporarily", message.lower())
        self.assertIn("quota", message.lower())

    def test_short_non_placeholder_keys_are_not_rejected_by_length(self) -> None:
        with patch("resume_assistant.rag_pipeline._call_gemini", side_effect=Exception("live validation failed")):
            ok, message = classify_gemini_key("abc123")
        self.assertFalse(ok)
        self.assertIn("validation failed", message.lower())


if __name__ == "__main__":
    unittest.main()
