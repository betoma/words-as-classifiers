import pickle

import matplotlib.pyplot as plt
import numpy as np

with open("data/fuzzy_sem_results", "rb") as f:
    vv = pickle.load(f)

n_vals = [x[0] for x in vv]
results = [
    ([x[1][0] for x in vv], [x[2][0] for x in vv]),
    ([x[1][1] for x in vv], [x[2][1] for x in vv]),
    ([x[1][2] for x in vv], [x[2][2] for x in vv]),
    ([x[1][3] for x in vv], [x[2][3] for x in vv]),
]
labels = ["Mean", "Median", "Maximum", "Minimum"]

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for ax, model, lab in zip(axes.flatten(), results, labels):
    t = model[0]
    f = model[1]
    ax.plot(n_vals, t, lw=5, alpha=0.8, label="True")
    ax.plot(n_vals, f, lw=5, alpha=0.8, label="False")
    ax.set(xlabel="Number of words per class in vocab", ylabel=f"{lab} predicted value")
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels, title="Gold label", loc="center", bbox_to_anchor=(0.51, 0.49)
)
plt.savefig("images/fuzzy_results.png", bbox_inches="tight")
