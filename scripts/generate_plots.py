import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            '{:.1%}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom")


def plot_rects(true_overlap, rand_overlap, savefile=None):
    true_count = Counter(true_overlap)
    rand_count = Counter(rand_overlap)

    true_percent = [true_count[(i, False)] / (true_count[(i, False)] + true_count[(i, True)])
                    for i in range(3)]
    rand_percent = [rand_count[(i, False)] / (rand_count[(i, False)] + rand_count[(i, True)])
                    for i in range(3)]
    x = np.arange(3)
    width = 0.35

    fig, ax = plt.subplots()
    rects_true = ax.bar(x - width / 2, true_percent, width=width, label='BPE')
    rects_rand = ax.bar(x + width / 2, rand_percent, width=width, label='Random')

    ax.set_xticks(x)
    ax.set_xticklabels(['Alpha Helix', 'Strand', 'Beta Sheet'])
    ax.set_ylabel('Percent Agreement')
    ax.set_title('Agreement between tokens and secondary structure labels')
    ax.legend(loc='lower right')

    autolabel(ax, rects_true)
    autolabel(ax, rects_rand)

    plt.tight_layout()

    plt.show()
