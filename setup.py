"""Setup configuration for HiReLC package."""

from pathlib import Path
import re

from setuptools import setup, find_packages


ROOT = Path(__file__).parent


def _read_version() -> str:
    init_path = ROOT / "hirelc_package" / "__init__.py"
    if not init_path.exists():
        return "0.0.0"
    content = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content, re.M)
    if match:
        return match.group(1)
    return "0.0.0"


def _read_readme() -> str:
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        return ""
    return readme_path.read_text(encoding="utf-8")


setup(
    name="hirelc",
    version=_read_version(),
    author="HiReLC Team",
    author_email="your.email@example.com",
    description="Hierarchical Reinforcement Learning for Model Compression",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hirelc",
    packages=find_packages(include=["hirelc_package", "hirelc_package.*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "timm>=0.6.0",
        "gymnasium>=0.27.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "hirelc=hirelc_package.cli:main",
        ],
    },
)
