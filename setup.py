from setuptools import setup, find_packages
import os

# Read the README file for the long description
readme_path = "README.md"
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="NebulaPy",  # Name of the package
    version="0.1.0",  # Initial version
    author="Vedant Kulkarni",  # Replace with your name
    author_email="vedantkulkarni208@gmail.com",  # Replace with your email
    description="A lightweight library to track and organize AI experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vedant-8/NebulaPy",  # Replace with the actual URL of your repository
    project_urls={
        "Documentation": "https://github.com/Vedant-8/NebulaPy/tree/main/docs",  # Update as needed
        "Source": "https://github.com/Vedant-8/NebulaPy",  # Repository link
        "Tracker": "https://github.com/Vedant-8/NebulaPy/issues",  # Issues page link
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "nebulaPy"},  # Points to the package directory
    packages=find_packages(where="nebulaPy"),  # Automatically find sub-packages
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3.1",  # For visualization.py
        "pandas>=1.0",  # For data handling
        "tensorflow>=2.5",  # TensorFlow integration
        "torch>=1.9",  # PyTorch integration
        "scikit-learn>=0.24",  # Scikit-learn integration
    ],
    extras_require={
        "dev": ["pytest>=7.0", "flake8"],  # Development dependencies
    },
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    entry_points={
        "console_scripts": [
            "nebulapy=nebulaPy.__main__:main",  # Example CLI entry point (optional)
        ]
    },
)
