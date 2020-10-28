import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chord_recognition",
    version="0.1",
    author="Alex Rychyk",
    author_email="odiscort@gmail.com",
    description="Chord recognition library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/discort/chord-recognition",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'librosa==0.8.0',
        'livelossplot==0.5.3',
        'pandas==1.0.3',
        'torch==1.4.0',
    ],
    package_data={'chord_recognition': ['models/*.pt']},
)
