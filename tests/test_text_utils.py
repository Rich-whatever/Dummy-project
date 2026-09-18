"""
Tests for ``text_utils.clip`` — the log-line truncation helper shared by
``workflow.graph`` and ``execution.worker``.

The end-to-end safety net mocks ``run_worker_task``, so it never executes the
worker's copy of this logic; this file covers it directly.
"""

from __future__ import annotations

import unittest

from text_utils import DEFAULT_LOG_LIMIT, clip


class TestClip(unittest.TestCase):
    def test_none_becomes_empty_string(self):
        self.assertEqual(clip(None), "")

    def test_empty_string_is_unchanged(self):
        self.assertEqual(clip(""), "")

    def test_short_text_is_unchanged(self):
        self.assertEqual(clip("hello"), "hello")

    def test_text_exactly_at_the_limit_is_unchanged(self):
        text = "x" * 50
        self.assertEqual(clip(text, limit=50), text)

    def test_over_limit_text_is_truncated_and_marked(self):
        text = "y" * 100
        result = clip(text, limit=10)
        self.assertTrue(result.startswith("y" * 10))
        self.assertIn("truncated: 100 chars, showing 10", result)

    def test_original_length_is_reported_not_the_kept_length(self):
        result = clip("z" * 1234, limit=5)
        self.assertIn("truncated: 1234 chars, showing 5", result)

    def test_non_string_input_is_coerced(self):
        self.assertEqual(clip(42), "42")
        self.assertIn("truncated: 4 chars, showing 2", clip(1234, limit=2))

    def test_default_limit_constant_is_used(self):
        text = "a" * (DEFAULT_LOG_LIMIT + 1)
        result = clip(text)
        self.assertIn(f"truncated: {DEFAULT_LOG_LIMIT + 1} chars", result)
        self.assertEqual(DEFAULT_LOG_LIMIT, 2000)


if __name__ == "__main__":
    unittest.main()
