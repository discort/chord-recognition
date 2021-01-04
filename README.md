# Chord Recognition

The tool to solve Audio Chord Recognition (Chord Estimation) [problem](https://www.music-ir.org/mirex/wiki/2020:Audio_Chord_Estimation).

Some of pre-trained model metrics of major and minor chords on [Isophonics](http://isophonics.net/datasets) and [Robbie Williams](https://www.researchgate.net/publication/260399240_Chord_and_Harmony_annotations_of_the_first_five_albums_by_Robbie_Williams) datasets.
_ | majmin | mirex |
--- |--- | --- |
beatles | 0.784 | 0.762 |
queen | 0.808 | 0.791 |
zweieck | 0.840 | 0.814 |
robbie_williams | 0.905 | 0.882 |

## Installation

    pip install git+https://github.com/discort/chord-recognition

## Usage

```python
from chord_recognition.utils import read_audio
from chord_recognition.predict import annotate_audio

audio_waveform, sr = read_audio('tests/fixtures/C_Am.mp3')
result = annotate_audio(audio_waveform, sr=sr, nonchord=True)
print(result)
[(0.0, 1.6718, 'N'),
 (1.6718, 3.3437, 'C'),
 (3.3437, 5.1084, 'N'),
 (5.1084, 6.0372, 'Am')]
```

## Development

#### Download datasets

Use this [link](https://drive.google.com/file/d/1diyRPrhuqphACRrni2_rrS5lbl0CRm2r/view?usp=sharing) to download Beatles, Queen and Robbie Williams datasets in [isophonics](http://www.isophonics.net/datasets) format

#### Unzip data and put into root of a project
    
    unzip data.zip data

#### Install requirements

    pip install -r requirements.txt

#### Run tests
    
    py.test -q --cov=chord_recognition tests

#### Experiments

```
experiments.ipynb  <- Describing some experiments to improve classification/segmentation
experiment<N>.ipynb  <- A code to reproduce an experiment
chords_analysis.ipynb  <- Check chord distribution and other stats data
audio_analysis.ipynb  <- Check spectrogram, chromagram, etc
```

#### Run jupyter

    jupyter notebook

References:
1. Müller M. (2015) [Chord Recognition. In: Fundamentals of Music Processing](https://doi.org/10.1007/978-3-319-21945-5_5)
2. Korzeniowski, Widmer (2016) [A Fully Convolutional Deep Auditory Model for Musical Chord Recognition](https://arxiv.org/abs/1612.05082)
3. Zanoni (2014) [Chord and Harmony annotations of the first five albums by Robbie Williams](https://www.researchgate.net/publication/260399240_Chord_and_Harmony_annotations_of_the_first_five_albums_by_Robbie_Williams)