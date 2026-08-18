"""
SciIF: Scientific Instruction Following Evaluation Framework

A comprehensive evaluation framework for assessing LLM performance on
scientific instruction following tasks with constraint validation.
"""

__version__ = "1.0.0"
__author__ = "SciIF Team"

from .evaluator import (
    main,
    load_problems,
    setup_logging,
)
from .api_client import run_responses_api, get_client
# LocalModelWrapper is optional (only needed for local model evaluation)
try:
    from .local_model import LocalModelWrapper
    __all__ = [
        "main",
        "load_problems",
        "setup_logging",
        "run_responses_api",
        "get_client",
        "LocalModelWrapper",
    ]
except (ImportError, SyntaxError, IndentationError):
    # If local_model has issues, just skip it
    __all__ = [
        "main",
        "load_problems",
        "setup_logging",
        "run_responses_api",
        "get_client",
    ]

