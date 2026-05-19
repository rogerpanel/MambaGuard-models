"""Setuptools shim for the MambaGuard reproducibility codebase.

Modern metadata is declared in `pyproject.toml`; this file exists for
backwards compatibility with `pip install -e .` on older toolchains and
to expose console entry points.
"""
from setuptools import find_packages, setup


def _read_requirements() -> list[str]:
    """Parse requirements.txt, skipping comments and blank lines."""
    reqs: list[str] = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


def _read_long_description() -> str:
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


setup(
    name="mambaguard",
    version="1.0.0",
    description=(
        "MambaGuard: Certified Selective State-Space Detection for "
        "Multi-Protocol LLM Agent Security."
    ),
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="Roger Nick Anaedevha, Alexander G. Trofimov",
    author_email="rnanaedevha@mephi.ru",
    url="https://github.com/rogerpanel/MambaGuard-models",
    project_urls={
        "Paper": "https://github.com/rogerpanel/MambaGuard-models",
        "Platform": "https://github.com/rogerpanel/robustidps.ai",
        "Zenodo": "https://doi.org/10.5281/zenodo.19129512",
    },
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "benchmarks"]),
    install_requires=_read_requirements(),
    extras_require={
        "dev": ["pytest", "ruff", "black", "mypy"],
        "wandb": ["wandb"],
    },
    entry_points={
        "console_scripts": [
            "mambaguard-train=scripts.train:main",
            "mambaguard-eval=scripts.evaluate:main",
            "mambaguard-certify=scripts.certify:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
