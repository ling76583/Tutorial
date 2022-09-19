
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


conll = torch.load('./data/conll.pt')


#pos regression
def build_posdataset(raw_dataset):
    X = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[],
         10:[], 11:[], 12:[]}
    y = []
    for sen in raw_dataset:
        y.extend(sen['pos_labels'])
        for i in range(13):
            layer = sen['hidden_states'][i]
            for index in sen['word_token_indices']:
                pooled = np.sum(layer.numpy()[index, :], 0)/len(index)
                X[i].append(pooled)
    for i in range(13):
        X[i] = np.asarray(X[i])
    return X, y


X_train, y_train = build_posdataset(conll['train'])
X_test, y_test = build_posdataset(conll['validation'])
posaccu_list = []
for i in range(13):
    clf = LogisticRegression(random_state=7, ).fit(X_train[i], y_train)
    predicted = clf.predict(X_test[i])
    f1 = f1_score(y_test, predicted, average='macro')
    posaccu_list.append(f1)
    print("f1: ")
    print(f1)
print(posaccu_list)
print('finish pos')


#ner regression
def build_nerdataset(raw_dataset):
    X = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[],
         10:[], 11:[], 12:[]}
    y = []
    for sen in raw_dataset:
        y.extend(sen['ner_labels'])
        for i in range(13):
            layer = sen['hidden_states'][i]
            for index in sen['word_token_indices']:
                pooled = np.sum(layer.numpy()[index, :], 0)/len(index)
                X[i].append(pooled)
    for i in range(13):
        X[i] = np.asarray(X[i])
    return X, y


X_train, y_train = build_nerdataset(conll['train'])
X_test, y_test = build_nerdataset(conll['validation'])
neraccu_list = []
for i in range(13):
    clf = LogisticRegression(random_state=7, ).fit(X_train[i], y_train)
    predicted = clf.predict(X_test[i])
    f1 = f1_score(y_test, predicted, average='macro')
    neraccu_list.append(f1)
    print("f1: ")
    print(f1)
print(neraccu_list)
print('finish ner')


semeval = torch.load('./data/semeval.pt')


#rel regression
def build_reldataset(raw_dataset):
    X_1 = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[],
         10:[], 11:[], 12:[]}
    X_2 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [],
           10: [], 11: [], 12: []}
    y = []
    for rel in raw_dataset:
        y.append(rel['rel_label'])
        for i in range(13):
            X_1[i].append(rel['entity1_representations'][i].numpy())
            X_2[i].append(rel['entity2_representations'][i].numpy())
    for i in range(13):
        X_1[i] = np.asarray(X_1[i])
        X_2[i] = np.asarray(X_2[i])
    return X_1, X_2, y



X_1train, X_2train, y_train = build_reldataset(semeval['train'])
X_1test, X_2test, y_test = build_reldataset(semeval['test'])
relaccu_list = []
for i in range(13):
    clf = LogisticRegression(random_state=7).fit(X_1train[i]-X_2train[i], y_train)
    predicted = clf.predict(X_1test[i]-X_2test[i])
    f1 = f1_score(y_test, predicted, average='macro')
    relaccu_list.append(f1)
    print("f1: ")
    print(f1)
print(relaccu_list)





