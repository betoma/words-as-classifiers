from depparse import DepParse


parse = DepParse()
nps = parse.get_NPs("There are some white cats next to the tree.")
print(nps)
