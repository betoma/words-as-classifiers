import stanza


class DepParse:
    def __init__(self):
        self.parser = stanza.Pipeline("en")
        self.raw_parse = None
        self.nps = []

    def parse(self, doc):
        self.raw_parse = self.parser(doc)
        return self.raw_parse

    def get_NPs(self):
        if not self.raw_parse:
            print("Raw document has not been parsed. Please parse a document first.")
            return
        else:
            if self.nps:
                self.nps = []
            sent_count = 0
            for sent in self.raw_parse:
                headnouns = [x for x in sent if x["deprel"] == "nsubj"]
                mods = []
                for word in headnouns:
                    modifiers = [x for x in sent if x["head"] == int(word["id"])]
                    mods.append(modifiers)
                # add stuff to add the nouns and their modifiers to self.nps
                # be sure to include indices
                sent_count += 1
