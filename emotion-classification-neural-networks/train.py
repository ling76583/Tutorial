# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils
import models
import argparse

import random
import time
start_time = time.time()

# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework
# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.

def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    ########## YOUR CODE HERE ##########
    # TODO: Given a model, data, and loss function, you should do the following:
    # TODO: 1) Loop through the whole train dataset performing batch optimization with the optimizer of your choice,
    # TODO: updating the model parameters with each batch (we suggest you use torch.optim.Adam to start);
    # TODO: 2) Each time you reach the end of the train dataset (one "epoch"), calculate the loss on the whole dev set;
    # TODO and 3) stop training and return the model once the development loss stops improving (called early stopping).
    # TODO: Make sure to print the dev set loss each epoch to stdout.

    previous_loss = 999
    for epoch in range(10):
        for batch in train_generator:
            for param in model.parameters():
                param.grad = None
            output = model(batch[0])
            loss = loss_fn(output, batch[1].long())
            loss.backward()
            optimizer.step()
        print("--- %s seconds ---" % (time.time() - start_time))

        dev_loss = 0
        with torch.no_grad():
            for batch in dev_generator:
                output = model(batch[0])
                dev_loss += loss_fn(output, batch[1].long()).item()
        print('Epoch: '+str(epoch+1)+' Dev loss: ', dev_loss)

        #stop training when the development loss stops improving
        if previous_loss - dev_loss < 10**(-15):
            break
        previous_loss = dev_loss

    return model

##extension-grading
def train_model_attention(encoder, decoder, loss_fn, encoder_optimizer, decoder_optimizer,
                          train_generator, dev_generator, max_length):

    for epoch in range(4):
        select_list = [random.randint(0, len(train_generator)) for num in range(20)]
        select = -1
        for batch in train_generator:
            select +=1
            if select not in select_list:
                continue
            for param in encoder.parameters():
                param.grad = None
            for param in decoder.parameters():
                param.grad = None

            for i in range(len(batch[0])):
                #train encoder
                sentence = batch[0][i]
                encoder_hidden = encoder.initHidden()
                encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
                for ei in range(len(sentence)):
                    encoder_output, encoder_hidden = encoder(sentence[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                #train decoder
                decoder_input = torch.tensor([[0]])
                decoder_hidden = encoder_hidden
                loss = 0
                for di in range(len(sentence)):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += loss_fn(decoder_output, batch[1][i].unsqueeze(0).long())
                    decoder_input = batch[1][i]

                encoder_optimizer.step()
                decoder_optimizer.step()
        print("--- %s seconds ---" % (time.time() - start_time))

        dev_loss = 0
        with torch.no_grad():
            select_list = [random.randint(0, len(dev_generator)) for num in range(5)]
            select = -1
            for batch in dev_generator:
                select += 1
                if select not in select_list:
                    continue

                for i in range(len(batch[0])):
                    sentence = batch[0][i]
                    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
                    for ei in range(len(sentence)):
                        encoder_output, encoder_hidden = encoder(sentence[ei], encoder_hidden)
                        encoder_outputs[ei] = encoder_output[0, 0]

                    decoder_input = torch.tensor([[0]])
                    decoder_hidden = encoder_hidden
                    for di in range(len(sentence)):
                        decoder_output, decoder_hidden, decoder_attention = decoder(
                            decoder_input, decoder_hidden, encoder_outputs)
                        dev_loss += loss_fn(decoder_output, batch[1][i].unsqueeze(0).long()).item()
                        decoder_input = batch[1][i]

        print('Epoch: ' + str(epoch + 1) + ' Dev loss: ', dev_loss)

    return encoder, encoder_hidden, decoder


def test_model(model, loss_fn, test_generator):
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


##extension-grading
def test_model_attention(encoder, encoder_hidden, decoder, loss_fn, test_generator, max_length):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    encoder.eval()
    decoder.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        select_list = [random.randint(0, len(test_generator)) for num in range(5)]
        select = -1
        for X_b, y_b in test_generator:
            select += 1
            if select not in select_list:
                continue
            # Predict
            for i in range(len(X_b)):
                sentence = X_b[i]
                y_pred = [0, 0, 0, 0]
                encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
                for ei in range(len(sentence)):
                    encoder_output, encoder_hidden = encoder(sentence[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]
                decoder_input = torch.tensor([[0]])
                decoder_hidden = encoder_hidden
                loss = 0
                for di in range(len(sentence)):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += loss_fn(decoder_output, y_b[i].unsqueeze(0).long()).item()
                    y_pred[decoder_output.argmax(1)] += 1
                    decoder_input = y_b[i]

                # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
                gold.append(y_b[i].cpu().detach().numpy())
                predicted.append(torch.tensor(y_pred).argmax().cpu().detach().numpy())

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time

        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev,
                                                                                                          test,
                                                                                                          BATCH_SIZE,
                                                                                                          EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    ########## YOUR CODE HERE ##########
    # TODO: for each of the two models, you should 1) create it,
    # TODO: 2) run train_model() to train it, and
    # TODO: 3) run test_model() on the result
    if args.model == 'dense':
        model = models.DenseNetwork(embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, loss_fn, optimizer, train_generator, dev_generator)
        test_model(model, loss_fn, test_generator)

    elif args.model == 'RNN':
        model = models.RecurrentNetwork(embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, loss_fn, optimizer, train_generator, dev_generator)
        test_model(model, loss_fn, test_generator)

    elif args.model == 'extension1':
        model = models.CNN(embeddings)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, loss_fn, optimizer, train_generator, dev_generator)
        test_model(model, loss_fn, test_generator)

    elif args.model == 'extension2':
        encoder = models.EncoderRNN(embeddings)
        decoder = models.AttnDecoderRNN()
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
        encoder, encoder_hidden, decoder = train_model_attention(encoder, decoder, loss_fn, encoder_optimizer, decoder_optimizer,
                                train_generator, dev_generator, 200)
        test_model_attention(encoder, encoder_hidden, decoder, loss_fn, test_generator, 200)

    else:
        None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "extension1", "extension2"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
