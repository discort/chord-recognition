import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler
import torch.nn.functional as F

from chord_recognition.dataset import ContextIterator, context_window
from chord_recognition.ann_utils import compute_annotation
from chord_recognition.utils import compute_chromagram, log_compression, \
    exponential_smoothing
from .cnn import model

CURR_DIR = os.path.dirname(__file__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(
    torch.load(os.path.join(CURR_DIR, 'models/etd_best_model.pt'), map_location=device))
model.eval()


@torch.no_grad()
def predict_annotations(spectrogram, model, device, batch_size=32, context_size=7, num_classes=25):
    criterion = nn.Softmax(dim=1)
    # ToDo:
    # - vectorize funtions (get frame matrix)
    frames = context_window(spectrogram, context_size)
    frames = np.asarray([f.reshape(1, *f.shape) for f in frames])

    sampler = BatchSampler(SequentialSampler(frames), batch_size=batch_size, drop_last=False)
    result = torch.zeros(spectrogram.shape[1], num_classes)
    for idx in sampler:
        inputs = torch.from_numpy(frames[idx]).to(device=device, dtype=torch.float32)
        scores = model(inputs)
        scores = criterion(scores)
        scores = scores.squeeze(3).squeeze(2)
        scores = batch_exponential_smoothing(scores, 0.1)
        result[idx, :] = scores

    preds = torch.argmax(result, 1)
    result = F.one_hot(preds, num_classes).t_()
    return result.data.numpy()


def batch_exponential_smoothing(x, alpha):
    batsches, _ = x.shape
    result = []
    for i in range(batsches):
        x_smooth = exponential_smoothing(x[i, :].numpy(), alpha)
        result.append(torch.from_numpy(x_smooth))
    return torch.stack(result)


def annotate_audio(audio_waveform, Fs, window_size=8192, hop_length=4096,
                   ext_minor=None, nonchord=True):
    """Calculate the annotation of specified audio file

    - calculates spectrogram
    - applies filterbanks
    - logarithmise the filtered magnitudes to compress the value range
    - compute annotation matrix
    - apply exponential smoothing
    - convert annotation matrix to basic ann representation

    Args:
        audio_waveform: Audio time series (np.ndarray [shape=(n,))
        Fs: Sampling rate (int)
        ext_minor: label is used for a minor chord ('m' by default)
        nonchord: include or exclude non-chord class (bool)

    Returns:
        annotation: annotated file in format [(start, end, label), ...]
    """
    chromagram = compute_chromagram(
        audio_waveform=audio_waveform,
        Fs=Fs,
        window_size=window_size,
        hop_length=hop_length)
    chromagram = log_compression(chromagram)
    ann_matrix = predict_annotations(chromagram, model, device, batch_size=8)
    annotations = compute_annotation(
        ann_matrix, hop_length, Fs, ext_minor=ext_minor, nonchord=nonchord)
    return annotations
