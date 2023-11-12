import re
from pathlib import Path

from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


setup(
    name="transcription-diff",
    version=re.search(r"__version__\s+=\s+\"(.*)\"", Path("transcription_diff/__init__.py").read_text()).group(1),
    description="Speech to transcription comparison",
    author="Corentin Jemine",
    author_email="corentin.jemine@gmail.com",
    packages=find_packages(),
    platforms="any",
    python_requires=">=3.5",
    install_requires=requirements,
    long_description=long_description,
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
