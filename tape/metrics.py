from typing import Sequence
import numpy as np
import scipy.stats

from .registry import registry


@registry.register_metric('mse')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


@registry.register_metric('mae')
def mean_absolute_error(target: Sequence[float],
                        prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy')
def accuracy(labels: Sequence[int],
             scores: Sequence[float]) -> float:
    label_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    predictions = np.argmax(scores_array, -1)
    return np.mean(label_array == predictions)


@registry.register_metric('sequence_accuracy')
def sequence_accuracy(labels: Sequence[Sequence[int]],
                      scores: Sequence[Sequence[float]]) -> float:
    correct = 0
    total = 0
    for label, score in zip(labels, scores):
        label_array = np.asarray(label)
        scores_array = np.asarray(score)
        predictions = np.argmax(scores_array, -1)
        mask = label_array != -1
        is_correct = label_array[mask] == predictions[mask]

        correct += is_correct.sum()
        total += is_correct.size

    return correct / total
