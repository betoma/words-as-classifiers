import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, uniform
import stanza
import torch
from tqdm import tqdm, trange

from utils.depparse import DepParse
from utils.fakedata import FakeWorld, FakeData
from utils.semparse import FuzzySem

torch.cuda.empty_cache()

a, b = 5, 1.3
a2, b2 = 5, 1.5
a3, b3 = 5, 2.2
a4, b4 = 4, 5

distributions = {
    "exact": beta(a, b),
    "close": beta(a2, b2),
    "ish": beta(a3, b3),
    "not": beta(a4, b4),
}

w = FakeWorld(dists=distributions)
s = FuzzySem()
d = FakeData(w, 200)

data = list(s.split(d))

fig, ax = plt.subplots()
ax.violinplot(data)
ax.yaxis.grid(True)
ax.set_xticks([y + 1 for y in range(len(data))])
ax.set_xlabel("Gold label")
ax.set_ylabel("Generated probability")

plt.setp(ax, xticks=[y + 1 for y in range(len(data))], xticklabels=["True", "False"])

plt.savefig("images/fuzzy_violin.png", bbox_inches="tight")
