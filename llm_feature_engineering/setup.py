from setuptools import setup, find_packages

setup(
    name="llm-feature-engineering",
    version="0.1.0",
    author="Edo",
    description="LLM-based feature engineering methods for machine learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "anthropic>=0.7.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.43.0",
        "lightgbm>=4.0.0",
        "featuretools>=1.28.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)