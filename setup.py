from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
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
        "Documentation": "https://github.com/Vedant-8/NebulaPy/docs",  # Replace with actual documentation link
        "Source": "https://github.com/Vedant-8/NebulaPy",  # Repository link
        "Tracker": "https://github.com/Vedant-8/NebulaPy/issues",  # Issues page link
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "nebulaPy"},  # Points to the package directory
    packages=find_packages(where="nebulaPy"),  # Automatically find sub-packages
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3.1",  # For visualization.py
        "pandas>=1.0",  # Replace/add other dependencies as needed
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
