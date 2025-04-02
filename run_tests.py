#!/usr/bin/env python
"""
Script to run tests for the AI Agent System.
"""

import os
import sys
import unittest
import argparse


def run_tests(verbose=False, pattern="test_*.py"):
    """
    Run tests for the AI Agent System.

    Args:
        verbose: Whether to show verbose output
        pattern: Pattern to match test files
    """
    # Set up test environment
    os.environ["ANTHROPIC_API_KEY"] = "mock_api_key"
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern=pattern)
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = test_runner.run(test_suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def main():
    """Main entry point for the test runner."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run tests for the AI Agent System")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )
    parser.add_argument(
        "-p", "--pattern", default="test_*.py", help="Pattern to match test files"
    )
    args = parser.parse_args()
    
    # Run tests
    sys.exit(run_tests(args.verbose, args.pattern))


if __name__ == "__main__":
    main()
