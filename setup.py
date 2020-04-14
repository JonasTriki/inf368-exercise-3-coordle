import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coordle", 
    version="0.0.1",
    author="Naphat Amundsen, Jonas Triki",
    author_email="naphat@live.no, trikijonas@gmail.com",
    description="Package for Coordle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonasTriki/inf368-exercise-3-coordle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17.2',
        'pandas>=0.25.1',
        'scispacy>=0.2.4',
        'scipy>=1.4.1',
        'numba>=0.45.1',
        'gensim>=3.8.1',
        'nltk>=3.4.5',
        'spacy-langdetect>=0.1.2',
    ],
)

# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz