import torch.nn as nn
import torch.nn.functional as F


class LSTMTagger(nn.Module):
    """ Actual Neural Network implementation class"""

    def __init__(self, embedding_dim, hidden_dim, wordspace_size, tagspace_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(wordspace_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.tag_from_hidden = nn.Linear(hidden_dim, tagspace_size)

    def forward(self, sentence):
        N = len(sentence)
        embeddings = self.word_embeddings(sentence)
        embeddings = embeddings.view(N, 1, -1)
        lstm_outs, _ = self.lstm(embeddings)
        lstm_outs = lstm_outs.view(N, -1)
        tags = self.tag_from_hidden(lstm_outs)
        tag_probs = F.log_softmax(tags, dim=1)
        return tag_probs
