import numpy as np
from num2words import num2words
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted

from .depparse import DepParse
from .fakedata import FakeData


class SemParse:
    numbers = {num2words(n): n for n in range(1, 20)}

    def __init__(self, parser=None):
        self.parser = parser

    def parse(self, dataset: FakeData):
        joined_data = "\n\n".join(dataset.utterances)
        np_sent_list = []
        depparser = DepParse.parse(joined_data, parser=self.parser)
        nps_per_sent = depparser.get_NPs()
        for noun_phrases in nps_per_sent:
            np_list = []
            for n in noun_phrases:
                head_classifier = n[1][0].lemma
                attr_classifier = None
                req_num = None
                for word in n[1][1]:
                    if word.upos == "ADJ":
                        attr_classifier = word.lemma
                    elif word.upos in {"DET", "NUM"}:
                        if word.lemma == "a" or word.lemma == "an":
                            req_num = 1
                        elif word.lemma in self.numbers:
                            req_num = self.numbers[word.lemma]
                np_list.append((n[0], (head_classifier, attr_classifier, req_num)))
            np_sent_list.append(np_list)
        return np_sent_list


class ClassicalSem(SemParse):
    def __init__(self, *args, threshold: float = 0.8, **kwargs):
        super(ClassicalSem, self).__init__(*args, **kwargs)
        self.threshold = threshold

    def set_threshold(self, new_threshold: float):
        self.threshold = new_threshold

    def classify(self, dataset: FakeData):
        parsed_sentences = self.parse(dataset)
        labels = []
        for i, sent in enumerate(parsed_sentences):
            per_sent = []
            classification_array = dataset.class_output[i] > self.threshold
            for n in sent:
                n_col_no = dataset.arr_cols[n[1][0]]
                n_class = classification_array[:, n_col_no]
                if n[1][1]:
                    a_col_no = dataset.arr_cols[n[1][1]]
                    a_class = classification_array[:, a_col_no]
                    output_val = n_class & a_class
                else:
                    output_val = n_class
                per_sent.append((output_val, n[1][2]))
            labels.append(per_sent)
        return labels

    def predict(self, dataset: FakeData, labels: """list of lists of labels""" = None):
        if not labels:
            labels = self.classify(dataset)
        return [int(all([z[0].sum() >= z[1] for z in x])) for x in labels]

    def score(self, dataset: FakeData, results: """list of bools""" = None, **kwargs):
        if not results:
            results = self.predict(dataset, **kwargs)
        return (
            accuracy_score(results, dataset.correct),
            f1_score(results, dataset.correct),
        )


if __name__ == "__main__":
    pass
