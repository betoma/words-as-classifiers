import numpy
from num2words import num2words
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from .depparse import DepParse


class SemParse(BaseEstimator):
    numbers = {num2words(n): n for n in range(1, 20)}

    def __init__(self, threshold: float = 0.8, parser=None):
        self.threshold = threshold
        self.np_sent_list = []
        self.X_ = []
        if parser:
            self.parser = parser
        else:
            self.parser = DepParse()

    def parse(
        self, joined_data: """sentences separated by two newlines all as one string"""
    ):
        nps_per_sent = self.parser.get_NPs(joined_data)
        for nps in nps_per_sent:
            np_list = []
            for np in nps:
                head_classifier = np[1][0].lemma
                attr_classifier = None
                req_num = None
                for word in np[1][1]:
                    if word.upos == "ADJ":
                        attr_classifier = word.lemma
                    elif word.upos in {"DET", "NUM"}:
                        if word.lemma == "a" or word.lemma == "an":
                            req_num = 1
                        elif word.lemma in self.numbers:
                            req_num = self.numbers[word.lemma]
                np_list.append((np[0], (head_classifier, attr_classifier, req_num)))
            self.np_sent_list.append(np_list)

    def fit(
        self,
        X: """FakeData.class_output""",
        y: """FakeData.correct""",
        joined_data: """FakeData.utterances joined by two newlines""" = None,
        arr_cols: """FakeData.arr_cols""" = None,
    ):
        """
        Takes the output of the classifier in dataset from FakeData class instance and applies a given threshold to determine each entity's binary value for the np from the corresponding sentence in self.np_list
        """
        self.parse(joined_data)
        self.X_ = []
        for i, sent in enumerate(self.np_sent_list):
            per_sent = []
            for np in sent:
                n_col_no = arr_cols[np[1][0]]
                n_arr = X[i][:, n_col_no]
                if np[1][1]:
                    a_col_no = arr_cols[np[1][1]]
                    a_arr = X[i][:, a_col_no]
                    n_arr = numpy.true_divide(numpy.add(n_arr, a_arr), 2)
                per_sent.append((n_arr > self.threshold, np[1][2]))
            self.X_.append(per_sent)
        return self

    def predict(self, X=None):
        """turn the binary sets produced in fit() into truth values based on the utterances in the data set"""
        check_is_fitted(self)
        return [int(all([z[0].sum() >= z[1] for z in x])) for x in self.X_]

    def score(self, X, y):
        return accuracy_score(self.predict(), y)


if __name__ == "__main__":
    pass
