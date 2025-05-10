from setuptools import setup, find_packages
import io

setup(
    name="resonfit",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "ipykernel>=6.0.0",  # For notebook examples
        ],
    },
    author="ydkim0119",
    author_email="guaolbux@gmail.com",
    description="A modular tool for fitting superconducting resonator data",
    long_description=io.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ydkim0119/ResonFit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)