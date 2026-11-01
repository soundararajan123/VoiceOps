"""Automated Unit and Integration Tests for VoiceOps 2026 Support Agent.
Author: Pearl Lakshmi (pearllakhsmi786)
Repository: https://github.com/pearllakhsmi786/VoiceOps-2026-Project
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_person6 import TestPerson6


class TestSupportAgent(TestPerson6):
    """Full test suite covering dataset parsing, adaptive RAG retrieval, business day calculations,
    caller verification gating, ticket escalation, speech normalization, and voice pipeline.
    """
    pass


if __name__ == '__main__':
    unittest.main()
