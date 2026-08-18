"""Basic tests for SciIF package."""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sciif import load_problems, setup_logging
from sciif.config import get_model_config, add_model_config


class TestBasic(unittest.TestCase):
    """Basic functionality tests."""
    
    def test_config(self):
        """Test configuration functions."""
        # Test adding a model config
        add_model_config("test-model", {
            "api_key": "test-key",
            "api_url": "https://test.com",
            "model": "test-model",
            "api_scheme": "boyue",
        })
        
        # Test getting model config
        config = get_model_config("test-model")
        self.assertEqual(config["api_key"], "test-key")
        
        # Test error for non-existent model
        with self.assertRaises(ValueError):
            get_model_config("non-existent-model")
    
    def test_load_problems_empty(self):
        """Test loading empty or non-existent file."""
        problems = load_problems("/non/existent/file.json")
        self.assertEqual(problems, [])
    
    def test_setup_logging(self):
        """Test logging setup."""
        setup_logging("info")
        # Should not raise any errors


if __name__ == "__main__":
    unittest.main()

