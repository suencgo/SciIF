"""Setup script for SciIF package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="sciif",
    version="1.0.0",
    description="Scientific Instruction Following Evaluation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SciIF Team",
    author_email="",
    url="https://github.com/yourusername/sciif",
    packages=find_packages(),
    install_requires=[
        "openai>=1.30.0",
        "tqdm>=4.65.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "local": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "peft>=0.5.0",
            "accelerate>=0.20.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "sciif-evaluate=sciif.evaluator:main",
            "sciif-test-local=sciif.local_model:main",
        ],
    },
)

