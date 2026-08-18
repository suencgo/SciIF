"""
Configuration module for SciIF

This module provides model configuration and API settings.
Loads configuration from /root/code/instruct_gen_v1/src/config.py if available,
otherwise uses environment variables or defaults.
"""

import os
import sys
from typing import Dict, Any, Optional

# Try to load configuration from instruct_gen_v1
try:
    sys.path.insert(0, '/root/code/instruct_gen_v1/src')
    from config import MODEL_MAP as INSTRUCT_GEN_MODEL_MAP, API_SCHEMES as INSTRUCT_GEN_API_SCHEMES
    # Use the configuration from instruct_gen_v1
    MODEL_MAP = INSTRUCT_GEN_MODEL_MAP.copy()
    API_SCHEMES = INSTRUCT_GEN_API_SCHEMES.copy()
    # Update API URL for boyue scheme
    API_SCHEMES["boyue"]["api_url"] = INSTRUCT_GEN_API_SCHEMES["boyue"]["api_url"]
    API_URL = API_SCHEMES["boyue"]["api_url"]
except (ImportError, ModuleNotFoundError, AttributeError):
    # Fallback to default configuration
    # API Schemes Configuration
    API_SCHEMES = {
        "boyue": {
            "api_type": "openai_sdk",
            "api_url": os.getenv("BOYUE_API_URL", "https://api.boyuerichdata.opensphereai.com/v1"),
            "verify_ssl": True,
        },
        "ailab": {
            "api_type": "direct_http",
            "api_url": os.getenv("AILAB_API_URL", ""),
            "api_key": os.getenv("AILAB_API_KEY", ""),
            "verify_ssl": False,
        },
    }

    # Default API URL
    API_URL = os.getenv("API_URL", API_SCHEMES["boyue"]["api_url"])

    # Model Configuration Map
    MODEL_MAP: Dict[str, Dict[str, Any]] = {
        "gpt-5": {
            "api_key": os.getenv("GPT5_API_KEY", ""),
            "api_url": os.getenv("GPT5_API_URL", API_URL),
            "model": "gpt-5",
            "api_scheme": "boyue",
            "use_chat_completions": False,
        },
        "gpt-5.1": {
            "api_key": os.getenv("GPT5_1_API_KEY", ""),
            "api_url": os.getenv("GPT5_1_API_URL", API_URL),
            "model": "gpt-5.1",
            "api_scheme": "boyue",
            "use_chat_completions": False,
        },
        "gpt-5.2": {
            "api_key": os.getenv("GPT5_2_API_KEY", ""),
            "api_url": os.getenv("GPT5_2_API_URL", API_URL),
            "model": "gpt-5.2",
            "api_scheme": "boyue",
            "use_chat_completions": False,
        },
        "gpt-4o": {
            "api_key": os.getenv("GPT4O_API_KEY", ""),
            "api_url": os.getenv("GPT4O_API_URL", API_URL),
            "model": "gpt-4o",
            "api_scheme": "boyue",
            "use_chat_completions": True,
        },
        "gemini-3": {
            "api_key": os.getenv("GEMINI3_API_KEY", ""),
            "api_url": os.getenv("GEMINI3_API_URL", API_URL),
            "model": "gemini-3-pro-preview-thinking",
            "api_scheme": "boyue",
            "use_chat_completions": True,
        },
        "gemini-3-flash": {
            "api_key": os.getenv("GEMINI3_FLASH_API_KEY", ""),
            "api_url": os.getenv("GEMINI3_FLASH_API_URL", API_URL),
            "model": "gemini-3-flash-preview-thinking",
            "api_scheme": "boyue",
            "use_chat_completions": True,
        },
        "deepseek-v4-pro": {
            "api_key": os.getenv("DEEPSEEK_V4_PRO_API_KEY", ""),
            "api_url": os.getenv("DEEPSEEK_V4_PRO_API_URL", API_URL),
            "model": "bailian/deepseek-v4-pro",
            "api_scheme": "boyue",
            "use_chat_completions": True,
        },
        "glm-5.1": {
            "api_key": os.getenv("GLM_5_1_API_KEY", ""),
            "api_url": os.getenv("GLM_5_1_API_URL", API_URL),
            "model": "glm-5.1",
            "api_scheme": "boyue",
            "use_chat_completions": True,
        },
        "gpt-5.4": {
            "api_key": os.getenv("GPT5_4_API_KEY", ""),
            "api_url": os.getenv("GPT5_4_API_URL", API_URL),
            "model": "gpt-5.4",
            "api_scheme": "boyue",
            "use_chat_completions": False,
        },
        "qwen3.5-397b": {
            "api_key": os.getenv("QWEN3_5_397B_API_KEY", ""),
            "api_url": os.getenv("QWEN3_5_397B_API_URL", API_URL),
            "model": "Qwen/Qwen3.5-397B-A17B",
            "api_scheme": "boyue",
            "use_chat_completions": True,
        },
        "qwen3.5-35b-a3b": {
            "api_key": os.getenv("QWEN3_5_35B_A3B_API_KEY", ""),
            "api_url": os.getenv("QWEN3_5_35B_A3B_API_URL", API_URL),
            "model": "Qwen/Qwen3.5-35B-A3B",
            "api_scheme": "boyue",
            "use_chat_completions": True,
        },
    }

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration, raising error if not found."""
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Model '{model_name}' not found in MODEL_MAP. "
            f"Please configure it in sciif/config.py or add it via environment variables."
        )
    return MODEL_MAP[model_name]

def add_model_config(model_name: str, config: Dict[str, Any]) -> None:
    """Add or update a model configuration."""
    MODEL_MAP[model_name] = config
