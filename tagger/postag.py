import enum

class POS(enum.IntEnum):
    """Class to implement a mapping from the tags to scalars"""
    ADJ = 0
    ADP = enum.auto()
    ADV = enum.auto()
    AUX = enum.auto()
    CCONJ = enum.auto()
    DET = enum.auto()
    INTJ = enum.auto()
    NOUN = enum.auto()
    NUM = enum.auto()
    PART = enum.auto()
    PRON = enum.auto()
    PROPN = enum.auto()
    PUNCT = enum.auto()
    SCONJ = enum.auto()
    SYM = enum.auto()
    VERB = enum.auto()
    X = enum.auto()


def from_upostag(tag_list):
    tags = []
    for i, tag in enumerate(tag_list):
        if tag in POS.__dict__.keys():
            tags.append(POS.__dict__[tag])
        else:
            tags.append(POS.X)
            print(f'unrecognized tag {tag} at index {i} in {tag_list}')

    return  tags