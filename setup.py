import re
from pathlib import Path

from setuptools import setup, find_packages


setup(
    name="transcription-diff",
    version=re.search(r"__version__\s+=\s+\"(.*)\"", Path("transcription_diff/__init__.py").read_text()).group(1),
    description="Speech to transcription comparison",
    long_description="A small python library to find differences between audio and transcriptions\n"
                     "https://github.com/CorentinJ/transcription-diff/",
    author="Corentin Jemine",
    author_email="corentin.jemine@gmail.com",
    packages=find_packages(),
    platforms="any",
    python_requires=">=3.7",
    install_requires=Path("requirements.txt").read_text("utf-8").splitlines(),
    tests_require=["pytest>=7.0.0"],
    long_description_content_type="text/markdown",
    url="https://github.com/CorentinJ/transcription-diff",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
