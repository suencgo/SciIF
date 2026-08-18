#!/usr/bin/env python3
"""
Command-line script for evaluating API models using SciIF framework.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sciif.evaluator import main

if __name__ == "__main__":
    main()

