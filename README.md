# Chord Recognition

The tools to solve Audio Chord Recognition (Chord Estimation) [problem](https://www.music-ir.org/mirex/wiki/2020:Audio_Chord_Estimation).

Some of pre-trained model metrics of major and minor chords made by [mir_eval](https://github.com/craffel/mir_eval) on [Isophonics](http://isophonics.net/datasets) and [Robbie Williams](https://www.researchgate.net/publication/260399240_Chord_and_Harmony_annotations_of_the_first_five_albums_by_Robbie_Williams) datasets.
_ | majmin | mirex |
--- |--- | --- |
beatles | 0.792 | 0.771 |
queen | 0.815 | 0.798 |
zweieck | 0.839 | 0.811 |
robbie_williams | 0.908 | 0.885 |

## How it works

`chord-recognition` takes an audio file in `mp3` format, and then represents it as waweform in `numpy.array`. It computes [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) to split the input into frames representing 1.5 seconds of the audio. It applies a simple [ConvNet](https://en.wikipedia.org/wiki/Convolutional_neural_network) to each frame to classify into 25 classes (12 minor, 12 major plus a non-chord class). Finally the result is enhanced by [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model) smoothing.

## Installation

    pip install git+https://github.com/discort/chord-recognition

## Usage

```python
from chord_recognition import estimate_chords

result = estimate_chords(audio_path='tests/fixtures/C_Am.mp3', nonchord=True)
print(result)
[(0.0, 1.6718, 'N'),
 (1.6718, 3.7152, 'C'),
 (3.7152, 4.2725, 'N'),
 (4.2725, 6.0372, 'Am')]
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

#### Evaluation

```
python -m chord_recognition.evaluate -ds <dataset_name>
```

#### Experiments

```
audio_analysis.ipynb  <- Check spectrogram, chromagram, etc
chords_analysis.ipynb  <- Check chord distribution and other stats data
experiments.ipynb  <- Describing some experiments to improve classification/
experiment<N>.ipynb  <- A code to reproduce an experiment
```

#### Run jupyter

    jupyter notebook

References:
1. Müller M. (2015) [Chord Recognition. In: Fundamentals of Music Processing](https://doi.org/10.1007/978-3-319-21945-5_5)
2. Korzeniowski, Widmer (2016) [A Fully Convolutional Deep Auditory Model for Musical Chord Recognition](https://arxiv.org/abs/1612.05082)
3. Zanoni (2014) [Chord and Harmony annotations of the first five albums by Robbie Williams](https://www.researchgate.net/publication/260399240_Chord_and_Harmony_annotations_of_the_first_five_albums_by_Robbie_Williams)