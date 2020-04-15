import csv
from pathlib import Path

import torch
from torch import optim, nn

from tagger.data import get_dataloader
from tagger.model import LSTMTagger
from tagger.postag import POS
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

def index_dataset(data, index={}, unknown=''):
    index[unknown] = 0
    for sentence, _ in data:
        for word in sentence:
            if word not in index.keys():
                index[word] = len(index)
    return index


def words_to_tensor(words, index, unknown=''):
    filtered = [w if w in index.keys() else unknown for w in words]
    ind = [index[w] for w in filtered]
    return torch.tensor(ind, dtype=torch.long)


def tags_to_tensor(tags):
    ind = [t.value for t in tags]
    return torch.tensor(ind, dtype=torch.long)


def tensor_to_tags(tensors):
    return [POS(t.item()) for t in tensors]


class Solver():
    """
    The class to maintain the learning process of the model
    It holds all the parameters and hyper-parameters, has methods, saves info and etc.
    """

    def __init__(self, conf, hypers, datapath):
        self.conf = conf

        self.datapath = datapath
        self.hypers = hypers

        self.index = index_dataset(self.train_split())

        self.wordspace_size = len(self.index.keys())
        self.tagspace_size = len(POS)
        self.model = LSTMTagger(hypers['E_DIM'], hypers['H_DIM'], self.wordspace_size, self.tagspace_size)
        self.loss_func = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=hypers['lr'])

        self.save_hypers(self.conf)

    def dataloader(self, split, shuffle):
        return get_dataloader(self.datapath, split, self.hypers['token_mode'], shuffle=shuffle)

    def train_split(self):
        return self.dataloader('train', True)

    def test_split(self):
        return self.dataloader('test', False)

    def train_epoch(self):
        loss = None
        for sentence, target_tags in self.train_split():
            self.model.zero_grad()
            model_inputs = words_to_tensor(sentence, self.index)
            model_tags = self.model(model_inputs)
            target_tags = torch.stack(target_tags).view(-1)

            loss = self.loss_func(model_tags, target_tags)
            loss.backward()
            self.optimizer.step()

        return loss

    def train(self):
        for epoch in range(1, 1 + self.hypers['epochs']):
            print(f'epoch {epoch} started...')
            loss = self.train_epoch()
            self.save_loss(self.conf, epoch, loss)

            print(f'LOSS = {loss}')
            print(f'epoch {epoch} completed')

        self.save_model(self.conf, self.hypers['epochs'])

    def evaluate(self):
        """
        During the evaluation we compute different metrics
        Sentence and token accuracy scores and confusion matrices
        for more analysis.
        """
        ground_truth, predictions = [], []

        correct_sentences = 0
        correct_tokens = 0

        all_sentences = 0
        all_tokens = 0

        with torch.no_grad():
            for sentence, target_tags in self.test_split():
                model_inputs = words_to_tensor(sentence, self.index)
                model_tags = self.model(model_inputs)
                _, predicted_tags = torch.max(model_tags, dim=1)

                target_tags = torch.stack(target_tags).view(-1)

                ground_truth.extend(tensor_to_tags(target_tags))
                predictions.extend(tensor_to_tags(predicted_tags))

                ct = 0
                at = 0
                for p, t in zip(predicted_tags, target_tags):
                    if p == t:
                        ct += 1
                    at += 1

                if ct == at:
                    correct_sentences += 1
                all_sentences += 1
                correct_tokens += ct
                all_tokens += at

        sentence_accuracy = correct_sentences / all_sentences
        token_accuracy = accuracy_score(ground_truth, predictions)

        labels = [p.name for p in POS]
        cm = confusion_matrix(ground_truth, predictions, normalize='true')
        cmd = ConfusionMatrixDisplay(cm, labels)
        fig, ax = plt.subplots(figsize=(15, 15))
        cmd.plot(include_values=True, xticks_rotation='vertical', values_format='.2%', cmap='cividis', ax=ax)

        plt.show()


        return [
            'sentence accuracy : {:.2f} %'.format(100 * sentence_accuracy),
            'token    accuracy : {:.2f} %'.format(100 * token_accuracy),
        ]

    def save_model(self, conf, epoch):
        print(f'Saving model at epoch {epoch} ...')
        model_path = Path('models').joinpath(f'{conf}_{epoch}')
        torch.save(self.model.state_dict(), model_path)

    def save_loss(self, conf, epoch, loss, digits=5):
        print(f'Saving LOSS CSV at epoch {epoch} ...')
        data = {'epoch': epoch, 'loss': '{:.{}f}'.format(loss, digits)}
        csv_p = Path('loss').joinpath(f'{conf}')

        once = not csv_p.exists()
        if once:
            csv_p.parent.mkdir(exist_ok=True)

        csv_file = open(csv_p, 'a', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=data.keys())

        if once:
            csv_writer.writeheader()

        csv_writer.writerow(data)
        csv_file.close()

    def save_hypers(self, conf):
        print(f'Saving hyper parameters at conf {conf} CSV ...')
        data = {'conf': conf}
        data.update(self.hypers)
        csv_p = Path('conf').joinpath('conf')

        once = not csv_p.exists()
        if once:
            csv_p.parent.mkdir(exist_ok=True)

        csv_file = open(csv_p, 'a', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=data.keys())

        if once:
            csv_writer.writeheader()

        csv_writer.writerow(data)
        csv_file.close()
