from setuptools import setup, find_packages

setup(
    name="resonfit",
    version="0.1.0",
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
            "isort>=5.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    author="ydkim0119",
    author_email="ydkim0119@example.com",  # Replace with your actual email
    description="A modular tool for fitting superconducting resonator data",
    long_description=open("README.md").read(),
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
