import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader

from chord_recognition.ann_utils import ChordModel, convert_label_sequence,\
    convert_annotation_segments
from chord_recognition.transformations import Rescale, TRAIN_MEAN, TRAIN_STD
from chord_recognition.utils import log_filtered_spectrogram, split_with_context,\
    read_audio, one_hot, batch_sequence

# @torch.no_grad()
# def forward(model, dataloader, device, num_classes, criterion=None):
#     if not criterion:
#         criterion = nn.Softmax(dim=1)

#     result = []
#     for inputs in dataloader:
#         inputs = inputs.to(device=device, dtype=torch.float32)
#         scores = model(inputs)
#         scores = criterion(scores)
#         scores = scores.squeeze(3).squeeze(2)
#         result.append(scores)

#     result = torch.cat(result)
#     return result


DEFAULT_SAMPLE_RATE = 44100


class UnsupportedSampleRate(Exception):
    """
    Chord Recognition pipeline does not work
    correctly with provided sampling rate
    """
    pass


class AudioProcessor:
    """
    Class that encapsulates audio processing for provided audiowave signals
    """

    def extract_features(self, audiowave):
        raise NotImplemented

    def split(self):
        """
        Split the features into sequence of frames
        """
        raise NotImplemented


class FrameContextProcessor(AudioProcessor):
    """
    Args:
        window_size: FFT window size
        hop_length: number audio of frames between STFT columns
    """
    pass


class FrameSeqProcessor(AudioProcessor):
    """
    Args:
        window_size: FFT window size
        hop_length: number audio of frames between STFT columns
    """

    def __init__(self,
                 window_size,
                 hop_length,
                 fmin=65,
                 fmax=2100,
                 seq_length=100):
        self.window_size = window_size
        self.hop_length = hop_length
        self.seq_length = seq_length
        self.fmin = fmin
        self.fmax = fmax

    def extract_features(self, audiowave, sr):
        spec = log_filtered_spectrogram(
            audio_waveform=audiowave,
            sr=sr,
            window_size=self.window_size,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            num_bands=24
        )
        frames = self.split(spec.T)
        return frames

    def split(self, spec):
        """
        Split the features into sequence of frames
        """
        frames = batch_sequence(spec, self.seq_length, last=True)
        return frames


class ChordRecognition:
    """
    Encapsulates the Chord Recognition Pipeline

    Args:
        ext_minor: label is used for a minor chord ('m' by default)
        nonchord: include or exclude non-chord class (bool)
        postprocessing: whether or not to use postprocessing.
            Availables options are `None` or `hmm`.
            if `hmm` - Hidden Markov Model postprocessing is used
    """

    def __init__(self,
                 audio_processor,
                 model,
                 ext_minor='m',
                 nonchord=True):
        self.audio_processor = audio_processor
        self.model = model
        self.model.eval()  # set model to evaluation mode
        self.batch_size = 1
        self.transform = Rescale(TRAIN_MEAN, TRAIN_STD)
        self.nonchord = True
        self.chord_model = ChordModel(ext_minor=ext_minor)
        self.fps = 10  # frames per seconds (sample_rate/hop_size)

    def process(self, audio_waveform, sr):
        """
        Args:
            audio_descriptor: file containing audio

        Returns:
            chords in [(start, end, label), ...] format
        """
        if sr != DEFAULT_SAMPLE_RATE:
            raise UnsupportedSampleRate(f"Sample rate: {sr} is not supported")

        features = self.extract_features(audio_waveform, sr)
        features = self.preprocess(features)
        labels = self.predict_labels(features)
        chords = self.decode_chords(labels)
        return chords

    def extract_features(self, audio_waveform, sr):
        """
        Extract features from a given audio_waveform

        Args:
            audio_waveform: Audio time series (np.ndarray [shape=(n,))
            sr: Sampling rate (int)

        Returns:
            features matrix, (np.ndarray [shape=(num_features, N))
        """
        features = self.audio_processor.extract_features(audio_waveform, sr)
        return features

    def preprocess(self, features):
        """
        Preprocess features sequence

        Args:
            features - list[(np.ndarray [shape=(num_features, N))]

        Returns:
            features list[(np.ndarray [shape=(num_features, N))]
        """
        features = [self.transform(f) for f in features]
        return features

    def predict_labels(self, features):
        """
        Predict chord labels from a features sequence

        Args:
            features - list[(np.ndarray [shape=(num_features, N))]

        Returns:
            encoded labels - np.ndarray [shape=(N,)]
        """
        dataloader = self._prepare_dataloader(features)
        labels = self.model.predict(dataloader)
        return labels

    def decode_chords(self, labels):
        """
        Decodes labels from one-hot to human-readable format

        Args:
            labels - argmax np.array [shape=(N,)]

        Returns:
            chords in [(start, end, label), ...] format
        """

        result_labels = self.chord_model.onehot_to_labels(labels)
        ann_seg = convert_label_sequence(result_labels)
        annotations = convert_annotation_segments(ann_seg, Fs=self.fps)

        # chords = convert_onehot_ann(
        #     ann_matrix=labels,
        #     hop_length=self.hop_length,
        #     sr=sr,
        #     ext_minor=self.ext_minor,
        #     nonchord=self.nonchord)
        return annotations

    def _prepare_dataloader(self, features):
        sampler = BatchSampler(
            sampler=SequentialSampler(features),
            batch_size=self.batch_size,
            drop_last=False)
        dataloader = DataLoader(
            dataset=features,
            #sampler=sampler,
            batch_size=self.batch_size)
        return dataloader

    # def predict_labels(self, features):
    #     """
    #     Predict chord labels from a feature-matrix

    #     Args:
    #         features - (np.ndarray [shape=(num_features, N))

    #     Returns:
    #         logits - np.ndarray [shape=(N, num_classes)]
    #     """
    #     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #     dataloader = self._prepare_dataloader(features)
    #     ann_matrix = forward(self.model, dataloader, device, self.num_classes)
    #     ann_matrix = ann_matrix.data.numpy()
    #     return ann_matrix

    # def process(self, audio_waveform, sr):
    #     """
    #     Args:
    #         audio_waveform: Audio time series (np.ndarray [shape=(n,))
    #         sr: Sampling rate (int)

    #     Returns:
    #         chords in [(start, end, label), ...] format
    #     """
    #     if sr != DEFAULT_SAMPLE_RATE:
    #         raise UnsupportedSampleRate(f"Sample rate: {sr} is not supported")

    #     features = self.extract_features(audio_waveform, sr)
    #     features = self.preprocess(features)
    #     logits = self.predict_labels(features)
    #     labels = self.postprocess(logits)
    #     chords = self.decode_chords(labels, sr)
    #     return chords

    # def postprocess(self, logits):
    #     """
    #     Postprocess logits to get one-hot repr of labels

    #     Args:
    #         logits np.ndarray [shape=(N, num_classes)]

    #     Returns:
    #         one-hot representation of labels, np.array [shape=(N, num_classes)]
    #     """
    #     if not self.postprocessing:
    #         preds = np.argmax(logits, 1)
    #         labels = one_hot(preds, self.num_classes)
    #     elif self.postprocessing == 'hmm':
    #         labels = postprocess_HMM(logits.T, p=0.22).T
    #     else:
    #         raise ValueError(f"Invalid param: {postprocessing} for postprocessing")
    #     return labels


def estimate_chords(audio_path,
                    duration=None,
                    ext_minor=None,
                    nonchord=True):
    """
    Estimate chords from given audio

    Args:
        audio_path: path to an audio file
        duration: only load up to this much audio (in seconds)
        ext_minor: suffix-label for minor chords ('m' by default)
        nonchord: include or exclude non-chord class (bool)

    Returns:
        chords in [(start, end, label), ...] format
    """
    from chord_recognition.models import deep_harmony

    audio_waveform, sr = read_audio(
        path=audio_path,
        duration=duration)
    audio_processor = FrameSeqProcessor(
        window_size=8192,
        hop_length=4410)
    model = deep_harmony(pretrained=True,
                         n_feats=105,
                         n_cnn_layers=3,
                         num_classes=26,
                         n_rnn_layers=3)
    recognition = ChordRecognition(
        audio_processor=audio_processor,
        model=model,
        ext_minor=ext_minor,
        nonchord=nonchord)
    chords = recognition.process(
        audio_waveform=audio_waveform,
        sr=sr)
    return chords
