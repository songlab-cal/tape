from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_dict(data: Dict[str, Dict[str, float]]):
    fig, ax = plt.subplots()

    labels = list(data.keys())

    num_plots = len(labels)
    x = np.arange(num_plots)
    width = 0.7 / num_plots

    rects = [ax.bar(x - wi)]
