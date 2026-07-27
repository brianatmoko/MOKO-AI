from setuptools import setup, find_packages

setup(
    name="moko-ai",
    version="0.1.0",
    description="MOKO AI — Autonomous Cognitive Operating System",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Brian Atmoko",
    url="https://github.com/brianatmoko/MOKO-AI",
    license="MIT",
    packages=find_packages(include=["moko_core", "moko_core.*"]),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "requests",
        "sympy",
        "psutil",
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "trl",
            "gguf",
            "faiss-cpu",
            "z3-solver",
            "uvicorn",
            "PyQt6",
            "arxiv",
        ],
        "finetune": [
            "torch>=2.0.0",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "trl",
            "bitsandbytes",
        ],
        "gui": [
            "PyQt6",
            "pygame",
        ],
    },
    entry_points={
        "console_scripts": [
            "moko=moko_core.moko_os:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
