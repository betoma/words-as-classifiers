import numpy as np
from num2words import num2words

from .depparse import DepParse


class SemParse:
    numbers = {num2words(n): n for n in range(1, 10)}

    def __init__(self):
        self.parser = DepParse()

    def parse(self, sentence: str):
        nps = self.parser.get_NPs(sentence)
        np_list = []
        for np in nps:
            head_classifier = np[2][0].lemma
            attr_classifier = None
            for word in np[2][1]:
                if word.upos == "ADJ":
                    attr_classifier = word.lemma
                elif word.upos in {"DET", "NUM"}:
                    if word.lemma == "a":
                        req_num = 1
                    elif word.lemma in self.numbers:
                        req_num = self.numbers[word.lemma]
            np_list.append((np[0], np[1], (head_classifier, attr_classifier, req_num)))
        return np_list

    @staticmethod
    def produce_set(
        np_list: """list of tuples of form (head noun, adjective, ...)""",
        class_results: np.ndarray,
        threshold: float,
    ):
        """
        Takes the output of .parse for a single np and returns the set of entities whose probability for being that NP is greater than threshold according to class_results
        """
        arr_cols = {
            "cat": 0,
            "dog": 1,
            "man": 2,
            "woman": 3,
            "stool": 4,
            "chair": 5,
            "white": 6,
            "black": 7,
            "brown": 8,
        }
        n_col_no = arr_cols[np_list[0]]
        n_arr = class_results[:, n_col_no]
        if np_list[1]:
            a_col_no = arr_cols[np_list[1]]
            a_arr = class_results[:, a_col_no]
            n_arr = np.true_divide(np.add(n_arr, a_arr), 2)
        return n_arr > threshold

    def evaluate_truth(self, utterance: str, class_results, threshold):
        """
        Evaluates whether a "There is..." statement is true based using class_results and threshold
        """
        req = self.parse(utterance)
        truth_vals = []
        for s in req:
            n_set = self.produce_set(s[2], class_results, threshold)
            if n_set.sum() >= s[2][2]:
                truth_vals.append(True)
            else:
                truth_vals.append(False)
        return truth_vals


if __name__ == "__main__":
    pass
