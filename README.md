# Chord Recognition Library

A library to solve Audio Chord Recognition (Chord Estimation) [problem](https://www.music-ir.org/mirex/wiki/2020:Audio_Chord_Estimation).

Some of pre-trained model metrics of major and minor chords on [Isophonics](http://isophonics.net/datasets) and [Robbie Williams](https://www.researchgate.net/publication/260399240_Chord_and_Harmony_annotations_of_the_first_five_albums_by_Robbie_Williams) datasets.
_ | Precision | Recall | F1-measure
--- |--- | --- | --- |
Isophonics | 0.74 | 0.95 | 0.81 |
Robbie Williams | 0.53 | 0.60 | 0.56 |

## Installation

    pip install git+https://github.com/discort/chord-recognition

## Usage

```python
from chord_recognition.utils import read_audio
from chord_recognition.predict import annotate_audio

audio_waveform, Fs = read_audio('tests/fixtures/C_Am.mp3')
result = annotate_audio(audio_waveform, Fs=Fs, nonchord=True)
print(result)
[(0.0, 2.1362, 'N'),
 (2.1362, 2.2291, 'G'),
 (2.2291, 2.8793, 'C'),
 (2.8793, 4.1796, 'N'),
 (4.1796, 5.0155, 'A'),
 (5.0155, 6.3158, 'N')]
```

## Development

#### Download datasets

Use this [link](https://drive.google.com/file/d/1t6MU6ZI-27e25mKYcFbM5H5oUQrst7nD/view?usp=sharing) to download Beatles, Queen and Robbie Williams datasets in [isophonics](http://www.isophonics.net/datasets) format

#### Unzip data and put into root of a project
    
    unzip data.zip data

#### Install requirements

    pip install -r requirements.txt

#### Run tests
    
    py.test -q --cov=chord_recognition tests

#### Notebook list

```
nn_training.ipynb            <- Notebook containing train/val code
evaluation.ipynb             <- For model evaluation
chords_distribution.ipynb    <- Check chord distribution and other stats data
```

#### Run jupyter

    jupyter notebook

References:
1. Müller M. (2015) [Chord Recognition. In: Fundamentals of Music Processing](https://doi.org/10.1007/978-3-319-21945-5_5)
2. Korzeniowski, Widmer (2016) [A Fully Convolutional Deep Auditory Model for Musical Chord Recognition](https://arxiv.org/abs/1612.05082)
3. Zanoni (2014) [Chord and Harmony annotations of the first five albums by Robbie Williams](https://www.researchgate.net/publication/260399240_Chord_and_Harmony_annotations_of_the_first_five_albums_by_Robbie_Williams)