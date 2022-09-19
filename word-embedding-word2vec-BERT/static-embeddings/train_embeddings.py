
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re, string
import multiprocessing

from gensim.models import Word2Vec

from scipy.sparse import lil_matrix, save_npz, load_npz, linalg
import math
import pandas as pd
import numpy as np

with open('./data/brown.txt') as f:
    brown_corpus = f.readlines()

brown_corpus = [text.lower().replace('\n', '') for text in brown_corpus]

regex = re.compile('[%s]' % re.escape(string.punctuation))
brown_corpus = [regex.sub('', text) for text in brown_corpus]

brown_corpus = [word_tokenize(text) for text in brown_corpus]

lemmatizer = WordNetLemmatizer()
brown_corpus = [[lemmatizer.lemmatize(word) for word in text] for text in brown_corpus]


# word2vec
for k in [50, 100, 300]:
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=10, sg=1, window=5, vector_size=k,
                         sample=1e-5, alpha=0.03, min_alpha=0.0007,
                         negative=10, workers=cores-1)
    w2v_model.build_vocab(brown_corpus, progress_per=10000)
    w2v_model.train(brown_corpus, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save("./model/brown_word2vec5_"+str(k)+".model")


# SVD
word_dict = {}
for text in brown_corpus:
    for i in range(len(text)):
        if text[i] not in word_dict:
            word_dict[text[i]] = len(word_dict)

window = 10
cooccurence_size = 0
pmi = {}
for text in brown_corpus:
    for i in range(len(text)):
        cooccurence_size += min(10, len(text)-i-1)
        for j in range(1, window+1):
            if i+j >= len(text):
                break
            if (text[i], text[i+j]) not in pmi:
                pmi[(text[i], text[i+j])] = [0]
            pmi[(text[i], text[i+j])][0] += 1
print('finish count')

word_count1 = {}
word_count2 = {}
for word_pair in pmi:
    if word_pair[0] not in word_count1:
        word_count1[word_pair[0]] = 0
    if word_pair[1] not in word_count2:
        word_count2[word_pair[1]] = 0
    word_count1[word_pair[0]] += pmi[word_pair][0]
    word_count2[word_pair[1]] += pmi[word_pair][0]
print('finish calculate')

pmi_df = pd.DataFrame(pmi).T.reset_index()
pmi_df[0] = pmi_df.apply(lambda row: math.log(row[0]*cooccurence_size/(word_count1[row.level_0]*word_count2[row.level_1])), axis=1)
print('finish pmi')

M = lil_matrix((len(word_dict), len(word_dict)), dtype=float)
for index, row in pmi_df.iterrows():
    if row[0] > 0:
        M[word_dict[row.level_0], word_dict[row.level_1]] = row[0]
M_coo = M.tocoo()
save_npz('./model/M_coo10.npz', M_coo)
print('pmi saved')

M_coo = load_npz('./model/M_coo10.npz')
for k in [50, 100, 300]:
    U, S, Vt = linalg.svds(M_coo, k=k)
    S_matrix = np.diag(S)
    W = np.matmul(U, np.sqrt(S_matrix))
    C = np.matmul(np.transpose(Vt), np.sqrt(S_matrix))
    word_list_for_append = [[word] for word in word_dict]
    W = np.append(word_list_for_append, W, axis=1)
    W = [' '.join(embedding) for embedding in W]

    with open('./model/W10_'+str(k)+'.txt', 'w') as f:
        for element in W:
            f.write(element + "\n")
        print('model saved')