from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NebulaPy",
    version="0.1.0",
    author="Vedant Kulkarni",
    author_email="vedantkulkarni208@gmail.com",
    description="A lightweight library to track and organize AI experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vedant-8/NebulaPy",
    project_urls={
        "Documentation": "https://github.com/Vedant-8/NebulaPy/tree/main/docs",
        "Source": "https://github.com/Vedant-8/NebulaPy",
        "Tracker": "https://github.com/Vedant-8/NebulaPy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "nebulaPy"},
    packages=find_packages(where="nebulaPy"),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3.1",
        "pandas>=1.0",
        "tensorflow>=2.5",
        "torch>=1.9",
        "scikit-learn>=0.24",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "flake8"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nebulapy=nebulaPy.__main__:main",
        ]
    },
)
