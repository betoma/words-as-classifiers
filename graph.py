import pickle
from random import getrandbits

import matplotlib.pyplot as plt
import numpy as np


with open("data/naive_sem_results", "rb") as f:
    vv = pickle.load(f)

with open("data/rand_results", "rb") as f:
    rvv = pickle.load(f)

thresholds = list(np.linspace(0.7, 0.9, 20))

fig, axes = plt.subplots(5, 2, figsize=(8, 13))
plt.subplots_adjust(hspace=0.5)
# fig.suptitle("Performance across thresholds for different numbers of confusors")

for ax, model, random in zip(axes.flatten(), vv, rvv):
    a = [y[1] for y in model[1]]
    ra = [y[1] for y in random[1]]
    ax.plot(thresholds, a, lw=5, alpha=0.8, label="semantic")
    ax.plot(thresholds, ra, lw=5, alpha=0.8, label="random")
    ax.set(title=f"{model[0]} Confusors", xlabel="Threshold value", ylabel="Accuracy")
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels, title="Prediction Method", loc="lower center",  # mode="expand",
)
plt.savefig("images/best_thresholds.png", bbox_inches="tight")
