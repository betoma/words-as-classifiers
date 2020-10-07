from collections import defaultdict
import torch
import stanza


# Note that this so far can only extact the NP subjects of "There is/are" sentences and that it cannot cope with even conjoined noun phrases. At this point, this is good enough for my purposes, but if I wanna extend this later I should fix that.
class DepParse:
    def __init__(self, parser=None):
        if parser:
            self.parser = parser
        else:
            self.parser = stanza.Pipeline(
                lang="en",
                processors={
                    "tokenize": "spacy",
                    "mwt": "default",
                    "pos": "default",
                    "lemma": "default",
                    "depparse": "default",
                },
            )
        self.clean_parse = []
        self.both_parses = []
        self.nps = []

    @classmethod
    def parse(cls, docs, **kwargs):
        p = cls(**kwargs)
        raw_parse = p.parser(docs)
        for sent in raw_parse.sentences:
            head_dict = defaultdict(list)
            for word in sent.words:
                head_dict[word.head].append(word)
            p.clean_parse.append(dict(head_dict))
        p.both_parses = zip(raw_parse.sentences, p.clean_parse)
        return p

    @staticmethod
    def get_children(sentence: """a sentence from within clean_parse""", index: int):
        children = []
        if index in sentence:
            children.extend(sentence[index])
        grandchildren = [
            DepParse.get_children(sentence, int(child.id)) for child in children
        ]
        grandchildren = [kid for branch in grandchildren for kid in branch]
        children.extend(grandchildren)
        return children

    def get_NPs(self):
        for sent in self.both_parses:
            sent_nouns = []
            headnouns = [word for word in sent[0].words if word.deprel == "nsubj"]
            mods = []
            for word in headnouns:
                if word.lemma.endswith("."):
                    word.lemma = word.lemma.rstrip(".")
                modifiers = self.get_children(sent[1], int(word.id))
                mods.append(modifiers)
            noun_phrases = zip(headnouns, mods)
            for pair in noun_phrases:
                flat_list = [pair[0]] + pair[1]
                s_list = sorted(flat_list, key=lambda k: int(k.id))
                np_string = " ".join([x.text for x in s_list])
                sent_nouns.append((np_string, pair))
            self.nps.append(sent_nouns)
        return self.nps
