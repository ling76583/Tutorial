import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np

class DenseNetwork(nn.Module):
    def __init__(self, embeddings):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.fc1 = nn.Linear(len(embeddings[0]), 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class
        x = self.embeddings(x)
        x = torch.sum(x, 1)
        x = nn.functional.relu(self.fc1(x.float()))
        x = self.fc2(x)

        return x


class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.input_size = 100
        self.hidden_size = 100
        self.layer_num = 2
        self.output_size = 4
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.layer_num, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        x = self.embeddings(x)
        x = torch.sum(x, 1)
        x = torch.unsqueeze(x, 0).float()
        hidden_params = torch.zeros(self.layer_num, x.size(0), self.hidden_size)
        x, hidden_params = self.rnn(x, hidden_params)
        x = x.contiguous().view(-1, self.hidden_size)
        x = self.fc(x)

        return x


##extension-grading
# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
#extension1: CNN
class CNN(nn.Module):
    def __init__(self, embeddings):
        super(CNN, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.kernel_num = 90
        self.kernel_sizes = [3, 4, 5]
        self.output_size = 4
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (K, 100)) for K in self.kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.output_size)

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        x = self.embeddings(x)
        x = torch.unsqueeze(x, 1).float()
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


##extension-grading
#extension2: RNN with attention - Encoder part
class EncoderRNN(nn.Module):
    def __init__(self, embeddings):
        super(EncoderRNN, self).__init__()
        self.hidden_size = 100
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):
        output = self.embeddings(input).view(1, 1, -1).float()
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


##extension-grading
#extension2: RNN with attention - Decoder part
class AttnDecoderRNN(nn.Module):
    def __init__(self):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = 100
        self.output_size = 4
        self.dropout_p = 0.1
        self.max_length = 200

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = nn.functional.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)