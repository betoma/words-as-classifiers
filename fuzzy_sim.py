from collections import defaultdict
import pickle
from random import getrandbits

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

p = stanza.Pipeline(
    lang="en",
    processors={
        "tokenize": "spacy",
        "mwt": "default",
        "pos": "default",
        "lemma": "default",
        "depparse": "default",
    },
)

examples_per_dataset = 150
values = []
for n in trange(1, 11):
    w = FakeWorld(n, n, dists=distributions)
    s = FuzzySem(parser=p)
    true_vals = defaultdict(list)
    false_vals = defaultdict(list)
    for _ in range(5):
        d = FakeData(w, examples_per_dataset)
        true, false = s.score(d)
        true_vals["mean"].append(true[0])
        true_vals["median"].append(true[1])
        true_vals["max"].append(true[2])
        true_vals["min"].append(true[3])
        false_vals["mean"].append(false[0])
        false_vals["median"].append(false[1])
        false_vals["max"].append(false[2])
        false_vals["min"].append(false[3])
    values.append((n, true_vals, false_vals))

vv = [
    (
        x[0],
        (
            np.mean(x[1]["mean"]),
            np.mean(x[1]["median"]),
            np.mean(x[1]["max"]),
            np.mean(x[1]["min"]),
        ),
        (
            np.mean(x[2]["mean"]),
            np.mean(x[2]["median"]),
            np.mean(x[2]["max"]),
            np.mean(x[2]["min"]),
        ),
    )
    for x in values
]

with open("data/fuzzy_sem_results", "wb") as f:
    pickle.dump(vv, f)
