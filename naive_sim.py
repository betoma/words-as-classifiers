from collections import Counter
import pickle
from random import getrandbits

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, uniform
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
import stanza
import torch
from tqdm import tqdm, trange

from utils.depparse import DepParse
from utils.fakedata import FakeWorld, FakeData
from utils.semparse import ClassicalSem

torch.cuda.empty_cache()

fig, ax = plt.subplots(1, 1)
a, b = 5, 1.3
a2, b2 = 5, 1.5
a3, b3 = 5, 2.2
a4, b4 = 4, 5

x = np.linspace(beta.ppf(0.01, 1, 1), beta.ppf(0.99, 1, 1), 100)
ax.plot(x, beta.pdf(x, a, b), "k-", lw=5, alpha=0.8)
ax.plot(x, beta.pdf(x, a2, b2), "b-", lw=5, alpha=0.6)
ax.plot(x, beta.pdf(x, a3, b3), "y-", lw=5, alpha=0.6)
ax.plot(x, beta.pdf(x, a4, b4), "r-", lw=5, alpha=0.6)
plt.savefig("images/classifier_distributions.png", bbox_inches="tight")

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

thresholds = list(np.linspace(0.7, 0.9, 20))
examples_per_dataset = 150
values = []
random_values = []
# ent_random_values = []
for n in trange(1, 11):
    vals = []
    rand_vals = []
    # ent_rand_vals = []
    w = FakeWorld(n, n, dists=distributions)
    s = ClassicalSem(parser=p)
    for t in tqdm(thresholds):
        acc_scores = []
        rand_acc_scores = []
        f1_scores = []
        rand_f1_scores = []
        s.set_threshold(t)
        for _ in range(5):
            d = FakeData(w, examples_per_dataset)
            results = s.score(d)
            acc = results[0]
            f1 = results[1]
            acc_scores.append(acc)
            f1_scores.append(f1)
            rand_r = [getrandbits(1) for _ in range(examples_per_dataset)]
            rand_acc, rand_f1 = s.score(d, results=rand_r)
            rand_acc_scores.append(rand_acc)
            rand_f1_scores.append(rand_f1)
        vals.append((t, acc_scores, f1_scores))
        rand_vals.append((t, rand_acc_scores, rand_f1_scores))
        # ent_rand_vals.append((t, ent_rand_acc_scores, ent_rand_f1_scores))
    values.append((n, vals))
    random_values.append((n, rand_vals))

vv = [(x[0], [(y[0], np.mean(y[1]), np.mean(y[2])) for y in x[1]]) for x in values]
rvv = [
    (x[0], [(y[0], np.mean(y[1]), np.mean(y[2])) for y in x[1]]) for x in random_values
]
# ervv = [
#     (x[0], [(y[0], np.mean(y[1]), np.mean(y[2])) for y in x[1]])
#     for x in ent_random_values
# ]

with open("data/naive_sem_results", "wb") as f:
    pickle.dump(vv, f)

with open("data/rand_results", "wb") as f:
    pickle.dump(rvv, f)

# with open("data/ent_rand_results", "wb") as f:
#     pickle.dump(ervv, f)
