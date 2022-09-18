

import argparse
import json
import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack

import nltk
#nltk.download('punkt')
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


###create a dataframe of Word Ngram representations at debate level
###first represent the pro side texts and the con side texts respectively by ngrams vectors (n = 1,2,3) weighed by tfidf scores
###then subtract the pro side representation by the cons side to get the representation for a debate
###cut down vocabulary size by filtering out ngrams with small or large document frequencies
def tfidf_feature(json_list, nrange):
    pro_text_list = []
    con_text_list = []
    for line in json_list:
        debate = json.loads(line)
        pro_text = []
        con_text = []
        for round in debate['rounds']:
            for content in round:
                if content['side'] == 'Pro':
                    pro_text.append(content['text'].lower())
                else:
                    con_text.append(content['text'].lower())
        pro_text_list.append(' '.join(pro_text))
        con_text_list.append(' '.join(con_text))

    countvectorizer = CountVectorizer(ngram_range=nrange, min_df=int(0.05*len(pro_text_list)), max_df=0.9)
    pro_ngram_count = countvectorizer.fit_transform(pro_text_list)
    pro_ngram_list = countvectorizer.get_feature_names()
    countvectorizer = CountVectorizer(ngram_range=nrange, min_df=int(0.05*len(con_text_list)), max_df=0.9)
    con_ngram_count = countvectorizer.fit_transform(con_text_list)
    con_ngram_list = countvectorizer.get_feature_names()

    tfidftransformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    pro_tf_idf = tfidftransformer.fit_transform(pro_ngram_count)
    pro_tf_idf = pro_tf_idf.toarray()
    pro_tfidf_df = pd.DataFrame(pro_tf_idf, columns=pro_ngram_list)
    tfidftransformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    con_tf_idf = tfidftransformer.fit_transform(con_ngram_count)
    con_tf_idf = con_tf_idf.toarray()
    con_tfidf_df = pd.DataFrame(con_tf_idf, columns=con_ngram_list)

    total_columns = list(set(pro_ngram_list+con_ngram_list))
    for col in total_columns:
        if col not in pro_ngram_list:
            pro_tfidf_df[col] = 0
        if col not in con_ngram_list:
            con_tfidf_df[col] = 0
    tfidf_df = pro_tfidf_df - con_tfidf_df

    return tfidf_df, total_columns


###create a dict of V/A/D scores at word level
def vad_vocab_dict():
    with open('./lexica/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt') as f:
        line_list = f.readlines()

    vad_dict = {}
    for line in line_list:
        vad_scores = line.strip().split('\t')
        vad_dict[vad_scores[0]] = [float(score) for score in vad_scores[1:]]

    return vad_dict


###calculate V/A/D scores for each side and then subtract the pro side scores by the cons side scores
def vad_feature(json_list, vad_dict):
    all_document_vad = []

    for line in json_list:
        debate = json.loads(line)
        pro_valence, pro_arousal, pro_dominance = 0, 0, 0
        con_valence, con_arousal, con_dominance = 0, 0, 0
        for round in debate['rounds']:
            for content in round:
                tokens = word_tokenize(content['text'].lower())
                if content['side'] == 'Pro':
                    for key in tokens:
                        if key in vad_dict.keys():
                            pro_valence += vad_dict[key][0]
                            pro_arousal += vad_dict[key][1]
                            pro_dominance += vad_dict[key][2]
                else:
                    for key in tokens:
                        if key in vad_dict.keys():
                            con_valence += vad_dict[key][0]
                            con_arousal += vad_dict[key][1]
                            con_dominance += vad_dict[key][2]

        all_document_vad.append([pro_valence-con_valence, pro_arousal-con_arousal, pro_dominance-con_dominance])

    return all_document_vad


###calculate text lengths, numbers of personal pronouns and exclamations as linguistic features
###compute features for each side and then subtract the pro side features by the con side features
def linguistic_feature(json_list):
    personal_pronouns = ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                         "my", "your", "his", "hers", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"]
    question_words = ['what', 'which', 'who', 'whom', 'where', 'why', 'when', 'how', 'whose']
    modal_verbs = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
    all_document_ling = []

    for line in json_list:
        debate = json.loads(line)
        pro_length, con_length = 0, 0
        pro_personal, con_personal = 0, 0
        #pro_question, con_question = 0, 0
        #pro_modal, con_modal = 0, 0
        pro_exclamation, con_exclamation = 0, 0
        #pro_number, con_number = 0, 0
        for round in debate['rounds']:
            for content in round:
                tokens = word_tokenize(content['text'].lower())
                if content['side'] == 'Pro':
                    pro_length += len(tokens)
                    pro_personal += len([w for w in tokens if w in personal_pronouns])
                    #pro_question += len([w for w in tokens if w in question_words])
                    #pro_modal += len([w for w in tokens if w in modal_verbs])
                    pro_exclamation = content['text'].count('!')
                    #pro_number += len(re.findall('\d+', content['text']))
                else:
                    con_length += len(tokens)
                    con_personal += len([w for w in tokens if w in personal_pronouns])
                    #con_question += len([w for w in tokens if w in question_words])
                    #con_modal += len([w for w in tokens if w in modal_verbs])
                    con_exclamation = content['text'].count('!')
                    #con_number += len(re.findall('\d+', content['text']))

        all_document_ling.append([pro_length-con_length, pro_personal-con_personal, pro_exclamation-con_exclamation])

    return all_document_ling


###basically calculate the similarities of voters' big issue opinions compared to the pro_debater's and the con_debater's
###and the similarities of voters' religious ideologies and political ideologies compared to the pro_debater's and the con_debater's
def user_feature(json_list, user_dict):
    all_document_user = []

    for line in json_list:
        debate = json.loads(line)

        pro_opinion_pattern = [0 for i in range(48)]
        pro_religious = ''
        pro_political = ''
        if debate['pro_debater'] in user_dict:
            pro_debater = user_dict[debate['pro_debater']]
            pro_opinion_pattern = [1 if opin == 'Pro' else -1 if opin == 'Con' else 0 for opin in pro_debater['big_issues_dict'].values()]
            if pro_debater['religious_ideology'] != 'Not Saying':
                pro_religious = pro_debater['religious_ideology']
            if pro_debater['political_ideology'] != 'Not Saying':
                pro_political = pro_debater['political_ideology']
        con_opinion_pattern = [0 for i in range(48)]
        con_religious = ''
        con_political = ''
        if debate['con_debater'] in user_dict:
            con_debater = user_dict[debate['con_debater']]
            con_opinion_pattern = [1 if opin == 'Pro' else -1 if opin == 'Con' else 0 for opin in con_debater['big_issues_dict'].values()]
            if con_debater['religious_ideology'] != 'Not Saying':
                con_religious = con_debater['religious_ideology']
            if pro_debater['political_ideology'] != 'Not Saying':
                con_political = con_debater['political_ideology']

        similarity_with_pro, similarity_with_con = 0, 0
        religious_similarity = 0
        political_similarity = 0
        for v in debate['voters']:
            if v not in user_dict:
                continue
            voter = user_dict[v]
            voter_opinion_pattern = [1 if opin == 'Pro' else -1 if opin == 'Con' else 0 for opin in voter['big_issues_dict'].values()]
            similarity_with_pro += sum([1 for i in range(48) if voter_opinion_pattern[i] == pro_opinion_pattern[i]])/48
            similarity_with_con += sum([1 for i in range(48) if voter_opinion_pattern[i] == con_opinion_pattern[i]])/48
            if voter['religious_ideology'] == pro_religious:
                religious_similarity +=1
            if voter['religious_ideology'] == con_religious:
                religious_similarity -=1
            if voter['political_ideology'] == pro_political:
                political_similarity +=1
            if voter['political_ideology'] == con_political:
                political_similarity -=1

        all_document_user.append([similarity_with_pro-similarity_with_con, religious_similarity, political_similarity])

    return all_document_user


###extract debate ids, categories and winners
def debate_category(json_list):
    all_document_info = []

    for line in json_list:
        debate = json.loads(line)
        all_document_info.append([debate["id"], debate['category'], debate['winner']])

    return all_document_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outputtrain', dest='outputtrain', required=True,
                        help='Full path to the file we will output the preprocessed training data')
    parser.add_argument('--outputtest', dest='outputtest', required=True,
                        help='Full path to the file we will output the preprocessed testing data')
    args = parser.parse_args()

    with open(args.train, 'r') as train_json_file:
        train_json_list = list(train_json_file)

    with open(args.test, 'r') as val_json_file:
        val_json_list = list(val_json_file)

    with open(args.user_data, 'r') as user_json_file:
        for file in user_json_file:
            user_dict = json.loads(file)

    ###construct training dataset
    nrange = (1, 3)
    tfidf_df, ngram_list = tfidf_feature(train_json_list, nrange)
    vad_dict = vad_vocab_dict()
    vad_df = pd.DataFrame(vad_feature(train_json_list, vad_dict), columns=['valence', 'arousal', 'dominance'])
    ling_df = pd.DataFrame(linguistic_feature(train_json_list), columns=['text_length', 'personal_num', 'exclamation_num'])
    user_df = pd.DataFrame(user_feature(train_json_list, user_dict), columns=['opinion_similarity', 'religious_similarity', 'political_similarity'])
    category_df = pd.DataFrame(debate_category(train_json_list), columns=['id', 'category', 'winner'])
    train_processed_df = pd.concat([tfidf_df, vad_df, ling_df, user_df, category_df], axis=1)
    train_processed_df.to_csv(args.outputtrain)

    ###construct testing dataset
    tfidf_df, ngram_list = tfidf_feature(val_json_list, nrange)
    vad_df = pd.DataFrame(vad_feature(val_json_list, vad_dict), columns=['valence', 'arousal', 'dominance'])
    ling_df = pd.DataFrame(linguistic_feature(val_json_list), columns=['text_length', 'personal_num', 'exclamation_num'])
    user_df = pd.DataFrame(user_feature(val_json_list, user_dict),columns=['opinion_similarity', 'religious_similarity', 'political_similarity'])
    category_df = pd.DataFrame(debate_category(val_json_list), columns=['id', 'category', 'winner'])
    train_processed_df = pd.concat([tfidf_df, vad_df, ling_df, user_df, category_df], axis=1)
    train_processed_df.to_csv(args.outputtest)


