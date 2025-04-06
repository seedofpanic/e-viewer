#!/usr/bin/env python3
"""
Main test runner for the screen-casting application.
Run this script to execute all tests.
"""

import unittest
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_threading_utils import TestThreadingUtils
from tests.test_translations import TestTranslations
from tests.test_hotkey_input import TestHotkeyInput
from tests.test_gemini_api import TestGeminiAPI, TestGoogleAIFileManager


def run_tests():
    """Run all test cases"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestThreadingUtils))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTranslations))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestHotkeyInput))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestGeminiAPI))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestGoogleAIFileManager))
    
    # Create test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = test_runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests()) 