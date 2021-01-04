import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
import torch.nn.functional as F

from chord_recognition.ann_utils import convert_onehot_ann
from chord_recognition.models import deep_auditory_v2
from chord_recognition.models import postprocess_HMM
from chord_recognition.utils import batch_exponential_smoothing,\
    log_filtered_spectrogram, Rescale, context_window


def _prepare_dataloader(spectrogram, context_size=7, transform=Rescale(), batch_size=32):
    frames = context_window(spectrogram, context_size)
    frames = np.asarray([transform(f).reshape(1, *f.shape) for f in frames])

    sampler = BatchSampler(
        sampler=SequentialSampler(frames),
        batch_size=batch_size,
        drop_last=False)
    dataloader = DataLoader(
        dataset=frames,
        sampler=sampler,
        batch_size=None)
    return dataloader


def predict_annotations(
        spectrogram,
        model,
        dataloader=None,
        num_classes=25,
        postprocessing='hmm'):
    """
    Predict annotations from a spectrogam using model

    Args:
        spectrogram - audio time series (np.ndarray [shape=(n,))
        model: torch.nn.Module classificator
        dataloader:
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not dataloader:
        dataloader = _prepare_dataloader(spectrogram)
    ann_matrix = forward(model, dataloader, device, num_classes)
    ann_matrix = ann_matrix.data.numpy()

    if not postprocessing:
        preds = torch.argmax(ann_matrix, 1)
        ann_matrix = F.one_hot(preds, num_classes)
    elif postprocessing == 'hmm':
        ann_matrix = postprocess_HMM(ann_matrix.T, p=0.22).T
    else:
        raise ValueError(f"Invalid param: {postprocessing} for postprocessing")
    return ann_matrix


@torch.no_grad()
def forward(model, dataloader, device, num_classes, criterion=None):
    if not criterion:
        criterion = nn.Softmax(dim=1)

    result = []
    for inputs in dataloader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        scores = model(inputs)
        scores = criterion(scores)
        scores = scores.squeeze(3).squeeze(2)
        #scores = batch_exponential_smoothing(scores, 0.1)
        result.append(scores)

    result = torch.cat(result)
    return result


def annotate_audio(audio_waveform, sr, window_size=8192, hop_length=4096,
                   ext_minor=None, nonchord=True):
    """Calculate the annotation of a specified audio file

    - calculates a spectrogram
    - applies filterbanks
    - logarithmise the filtered magnitudes to compress the value range
    - compute annotation matrix
    - convert annotation matrix to basic ann representation

    Args:
        audio_waveform: Audio time series (np.ndarray [shape=(n,))
        sr: Sampling rate (int)
        window_size: FFT window size
        hop_length: number audio of frames between STFT columns
        ext_minor: label is used for a minor chord ('m' by default)
        nonchord: include or exclude non-chord class (bool)

    Returns:
        annotation: annotated file in format [(start, end, label), ...]
    """
    model = deep_auditory_v2(pretrained=True)
    model.eval()  # set model to evaluation mode

    spec = log_filtered_spectrogram(
        audio_waveform=audio_waveform,
        sr=sr,
        window_size=window_size,
        hop_length=hop_length,
        fmin=65, fmax=2100, num_bands=24
    )
    ann_matrix = predict_annotations(spec, model)
    annotations = convert_onehot_ann(
        ann_matrix=ann_matrix,
        hop_length=hop_length,
        sr=sr,
        ext_minor=ext_minor,
        nonchord=nonchord)
    return annotations
