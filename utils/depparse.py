from collections import defaultdict
import stanza


# Note that this so far can only extact the NP subjects of "There is/are" sentences and that it cannot cope with even conjoined noun phrases. At this point, this is good enough for my purposes, but if I wanna extend this later I should fix that.
class DepParse:
    def __init__(self, parser=None):
        if parser:
            self.parser = parser
        else:
            self.parser = stanza.Pipeline("en")
        self.raw_parse = None
        self.clean_parse = []
        self.both_parses = None
        self.nps = []

    def parse(self, doc):
        self.raw_parse = self.parser(doc)
        self.clean_parse = []
        for sent in self.raw_parse.sentences:
            head_dict = defaultdict(list)
            for word in sent.words:
                head_dict[word.head].append(word)
            self.clean_parse.append(dict(head_dict))
        self.both_parses = zip(self.raw_parse.sentences, self.clean_parse)
        return self.both_parses

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

    def get_NPs(self, doc):
        if not self.raw_parse:
            self.parse(doc)
        if self.nps:
            self.nps = []
        sent_count = 0
        for sent in self.both_parses:
            headnouns = [word for word in sent[0].words if word.deprel == "nsubj"]
            mods = []
            for word in headnouns:
                modifiers = self.get_children(sent[1], int(word.id))
                mods.append(modifiers)
            noun_phrases = zip(headnouns, mods)
            for pair in noun_phrases:
                flat_list = [pair[0]] + pair[1]
                s_list = sorted(flat_list, key=lambda k: int(k.id))
                np_string = " ".join([x.text for x in s_list])
                self.nps.append((sent_count, np_string, pair))
            sent_count += 1
        return self.nps
