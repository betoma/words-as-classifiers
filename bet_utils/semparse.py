# import math

# import numpy as np
# from num2words import num2words
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.utils.validation import check_is_fitted

# from .depparse import DepParse
# from .fakedata import FakeData


def prob_at_least(p_e: """list of probabilities""", m: int):
    e = len(p_e)
    p = [[0 for j in range(e + 1)] for i in range(e + 1)]
    p[0][0] = 1
    for i in range(1, e + 1):
        p[i][0] = p[i - 1][0] * (1 - p_e[i - 1])
        for j in range(1, e + 1):
            p[i][j] = p[i - 1][j] * (1 - p_e[i - 1]) + p[i - 1][j - 1] * p_e[i - 1]
    ans = 0.0
    for j in range(m, e + 1):
        ans += p[e][j]
    return ans


# class SemParse:
#     numbers = {num2words(n): n for n in range(1, 20)}

#     def __init__(self, parser=None):
#         self.parser = parser

#     def parse(self, dataset: FakeData):
#         joined_data = "\n\n".join(dataset.utterances)
#         np_sent_list = []
#         depparser = DepParse.parse(joined_data, parser=self.parser)
#         nps_per_sent = depparser.get_NPs()
#         for noun_phrases in nps_per_sent:
#             np_list = []
#             for n in noun_phrases:
#                 head_classifier = n[1][0].lemma
#                 attr_classifier = None
#                 req_num = None
#                 for word in n[1][1]:
#                     if word.upos == "ADJ":
#                         attr_classifier = word.lemma
#                     elif word.upos in {"DET", "NUM"}:
#                         if word.lemma == "a" or word.lemma == "an":
#                             req_num = 1
#                         elif word.lemma in self.numbers:
#                             req_num = self.numbers[word.lemma]
#                 np_list.append((n[0], (head_classifier, attr_classifier, req_num)))
#             np_sent_list.append(np_list)
#         return np_sent_list


# class ClassicalSem(SemParse):
#     def __init__(self, *args, threshold: float = 0.8, **kwargs):
#         super(ClassicalSem, self).__init__(*args, **kwargs)
#         self.threshold = threshold

#     def set_threshold(self, new_threshold: float):
#         self.threshold = new_threshold

#     def classify(self, dataset: FakeData):
#         parsed_sentences = self.parse(dataset)
#         labels = []
#         for i, sent in enumerate(parsed_sentences):
#             per_sent = []
#             classification_array = dataset.class_output[i] > self.threshold
#             for n in sent:
#                 n_col_no = dataset.arr_cols[n[1][0]]
#                 n_class = classification_array[:, n_col_no]
#                 if n[1][1]:
#                     a_col_no = dataset.arr_cols[n[1][1]]
#                     a_class = classification_array[:, a_col_no]
#                     output_val = n_class & a_class
#                 else:
#                     output_val = n_class
#                 per_sent.append((output_val, n[1][2]))
#             labels.append(per_sent)
#         return labels

#     def predict(self, dataset: FakeData, labels: """list of lists of labels""" = None):
#         if not labels:
#             labels = self.classify(dataset)
#         return [int(all([z[0].sum() >= z[1] for z in x])) for x in labels]

#     def score(self, dataset: FakeData, results: """list of bools""" = None, **kwargs):
#         if not results:
#             results = self.predict(dataset, **kwargs)
#         return (
#             accuracy_score(results, dataset.correct),
#             f1_score(results, dataset.correct),
#         )


# class FuzzySem(SemParse):
#     def __init__(self, *args, **kwargs):
#         super(FuzzySem, self).__init__(*args, **kwargs)

#     def predict(self, dataset: FakeData):
#         parsed_sentences = self.parse(dataset)
#         labels = []
#         for i, sent in enumerate(parsed_sentences):
#             per_sent = []
#             for n in sent:
#                 n_col_no = dataset.arr_cols[n[1][0]]
#                 output = dataset.class_output[i][:, n_col_no]
#                 if n[1][1]:
#                     a_col_no = dataset.arr_cols[n[1][1]]
#                     a_class = dataset.class_output[i][:, a_col_no]
#                     output = np.multiply(output, a_class)
#                 per_sent.append((output, n[1][2]))
#             labels.append(per_sent)
#         return [math.prod([prob_at_least(z[0], z[1]) for z in x]) for x in labels]

#     def binary_score(
#         self,
#         dataset: FakeData,
#         threshold: float,
#         results: """list of floats between 0.0 and 1.0""" = None,
#         **kwargs
#     ):
#         if not results:
#             results = self.predict(dataset, **kwargs)
#         binary_class = [int(x >= threshold) for x in results]
#         return (
#             accuracy_score(binary_class, dataset.correct),
#             f1_score(binary_class, dataset.correct),
#         )

#     def split(
#         self,
#         dataset: FakeData,
#         results: """list of floats between 0.0 and 1.0""" = None,
#         **kwargs
#     ):
#         if not results:
#             results = self.predict(dataset, **kwargs)
#         true_vals = []
#         false_vals = []
#         for i, r in enumerate(results):
#             if dataset.correct[i] == 1:
#                 true_vals.append(r)
#             else:
#                 false_vals.append(r)
#         return true_vals, false_vals

#     def score(
#         self,
#         *args,
#         true_vals: """list of floats""" = None,
#         false_vals: """list of floats""" = None,
#         **kwargs
#     ):
#         if not (true_vals and false_vals):
#             true_vals, false_vals = self.split(*args, **kwargs)
#         return (
#             (np.mean(true_vals), np.median(true_vals), max(true_vals), min(true_vals)),
#             (
#                 np.mean(false_vals),
#                 np.median(false_vals),
#                 max(false_vals),
#                 min(false_vals),
#             ),
#         )


if __name__ == "__main__":
    pass
