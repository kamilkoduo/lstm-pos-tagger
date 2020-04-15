from conllu import parse_incr
from torch.utils.data import Dataset, DataLoader

from tagger.postag import from_upostag

DATA_PATHS = {
    'Taiga': {
        'train': 'data/ud-treebanks-v2.5/UD_Russian-Taiga/ru_taiga-ud-train.conllu',
        'test': 'data/ud-treebanks-v2.5/UD_Russian-Taiga/ru_taiga-ud-test.conllu',
    },
    'SynTag': {
        'train': 'data/ud-treebanks-v2.5/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu',
        'test': 'data/ud-treebanks-v2.5/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu',
    },
    'ENG_EWT': {
        'train': 'data/ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-train.conllu',
        'test': 'data/ud-treebanks-v2.5/UD_English-EWT/en_ewt-ud-test.conllu',
    },
}


def parse_sequence(seq, token_mode, tag_mode='upostag'):
    """Parses a single sequence of data"""
    tokens = []
    tags = []
    for token in seq.tokens:
        # here we ignore the invalid data which is there accidentally
        if token[tag_mode] == '_' and token[token_mode] == '_':
            continue

        tokens.append(token[token_mode].lower())
        tags.append(token[tag_mode])

    return tokens, from_upostag(tags)


class POSDataset(Dataset):
    """
    Class which extends torchvision Dataset.
    By convention it is the parent class for all PyTorch Dataset.
    All the data is read in RAM simultaneously.
    """

    def __init__(self, path, split, token_mode):
        self.path = path
        self.split = split
        self.token_mode = token_mode

        self.data = None

        self.load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load(self):
        self.data = []
        data_file = open(DATA_PATHS[self.path][self.split], "r", encoding="utf-8")
        sequence_gen = parse_incr(data_file)
        for seq in sequence_gen:
            self.data.append(parse_sequence(seq, token_mode=self.token_mode))


def get_dataloader(path, split, token_mode, shuffle=True, num_workers=2):
    """
    Presents a data loader for the dataset we have written above
    Here we do not want to wark with batches, that is why its size is 1
    """
    data_source = POSDataset(path, split, token_mode)

    data_loader = DataLoader(data_source, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    return data_loader


# just some code to check the file
# if __name__ == '__main__':
#     dl = get_dataloader('Taiga', 'train', 'form', shuffle=True)
#
#     for x in dl:
#         print(x)
