import stanza

nlp = stanza.Pipeline("en")

doc = nlp(
    "There are three white cats and two brown dogs. There is a man. There are many white women."
)

print(doc)
