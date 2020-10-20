# Chord Recognition Tools

Jupyter notebooks and library to solve Audio Chord Recognition (Chord Estimation) [problem](https://www.music-ir.org/mirex/wiki/2020:Audio_Chord_Estimation).

#### For using pre-trained model install by

    pip install git+https://github.com/discort/chord-recognition

## To train model manually

#### Download datasets

Use this [link](https://drive.google.com/file/d/1t6MU6ZI-27e25mKYcFbM5H5oUQrst7nD/view?usp=sharing) to download Beatles, Queen and Robbie Williams datasets in [isophonics](http://www.isophonics.net/datasets) format

#### Unzip data and put into root of a project
    
    unzip data.zip data

#### Install requirements

    pip install -r requirements.txt

#### Run tests
    
    py.test -s

#### Run jupyter

    jupyter notebook

References:
1. Müller M. (2015) [Chord Recognition. In: Fundamentals of Music Processing](https://doi.org/10.1007/978-3-319-21945-5_5)
2. Korzeniowski, Widmer (2016) [A Fully Convolutional Deep Auditory Model for Musical Chord Recognition](https://arxiv.org/abs/1612.05082)
3. Zanoni (2014) [Chord and Harmony annotations of the first five albums by Robbie Williams](https://www.researchgate.net/publication/260399240_Chord_and_Harmony_annotations_of_the_first_five_albums_by_Robbie_Williams)