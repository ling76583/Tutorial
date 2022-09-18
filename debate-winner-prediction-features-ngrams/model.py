
import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


###ngram feature selection by correaltions with annotated winners
###model specified features (in column_fix) are not subject to this selection
def feature_select(tfidf_feature, winner_list, select_num, column_fix):
    corr_list = tfidf_feature.corrwith(winner_list)
    corr_df = pd.DataFrame()
    corr_df['ngram'] = tfidf_feature.columns
    corr_df['corr'] = corr_list
    corr_df['corr'] = corr_df['corr'].abs()
    corr_df.sort_values(by='corr', inplace=True, ascending=False)
    select_col = corr_df['ngram'].tolist()[:select_num]
    return tfidf_feature[list(set(select_col+column_fix))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outfile', dest='outfile', required=True,
                        help='Full path to the file we will write the model predictions')
    args = parser.parse_args()

    ###specify columns that are included or not for each model
    drop_dict = {
        "Ngram": ['valence', 'arousal', 'dominance', 'text_length', 'personal_num', 'exclamation_num', 'opinion_similarity', 'religious_similarity', 'political_similarity'],
        "Ngram+Lex": ['text_length', 'personal_num', 'exclamation_num', 'opinion_similarity', 'religious_similarity', 'political_similarity'],
        "Ngram+Lex+Ling": ['opinion_similarity', 'religious_similarity', 'political_similarity'], "Ngram+Lex+Ling+User": []}
    column_drop = drop_dict[args.model]
    fix_dict = {
        "Ngram": [], "Ngram+Lex": ['arousal'],
        "Ngram+Lex+Ling": ['arousal', 'dominance', 'exclamation_num', 'personal_num'],
        "Ngram+Lex+Ling+User": ['arousal', 'dominance', 'exclamation_num', 'personal_num', 'opinion_similarity', 'religious_similarity', 'political_similarity']}
    column_fix = fix_dict[args.model]

    ###preapare the training set, feature normalization
    train_df = pd.read_csv(args.train, sep=',')
    #, error_bad_lines=False
    train_df = train_df[[col for col in train_df.columns if col not in column_drop]]
    train_y = train_df['winner'].apply(lambda x: 1 if x == 'Pro' else 0)
    for col in column_fix:
        if col in ['valence', 'arousal', 'dominance', 'text_length', 'personal_num', 'exclamation_num',
                   'opinion_similarity', 'religious_similarity', 'political_similarity']:
            train_df[col] = preprocessing.normalize([np.array(train_df[col])])[0]
    train_df.drop(['Unnamed:0', 'id', 'category', 'winner'], axis=1, inplace=True)
    train_df = feature_select(train_df, train_y, int(0.80*len(train_df.columns)), column_fix)

    ###preapare the training set, feature normalization
    test_df = pd.read_csv(args.test, sep=',')
    test_id = test_df['id']
    test_category = test_df['category']
    test_y = test_df['winner'].apply(lambda x: 1 if x == 'Pro' else 0)
    for col in column_fix:
        if col in ['valence', 'arousal', 'dominance', 'text_length', 'personal_num', 'exclamation_num',
                   'opinion_similarity', 'religious_similarity', 'political_similarity']:
            test_df[col] = preprocessing.normalize([np.array(test_df[col])])[0]
    test_df = test_df.drop(['Unnamed:0','id', 'winner', 'category'], axis=1)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df[train_df.columns]

    ###training and predicting
    clf = LogisticRegression(random_state=0).fit(train_df, train_y)
    pred_y = clf.predict(test_df)

    ###output testing performance
    print('majority baseline')
    total_accuracy = len([y for y in test_y if y == 1])/len(test_y)
    total_accuracy = max(total_accuracy, 1-total_accuracy)
    print('total performance', total_accuracy)
    religion_accuracy = len([test_y[i] for i in range(len(test_y)) if test_category[i] == 'Religion' and test_y[i] == 1]) / \
                        len([test_y[i] for i in range(len(test_y)) if test_category[i] == 'Religion'])
    religion_accuracy = max(religion_accuracy, 1 - religion_accuracy)
    print('religious performance', religion_accuracy)
    nonreligion_accuracy = len([test_y[i] for i in range(len(test_y)) if test_category[i] != 'Religion' and test_y[i] == 1]) / \
                        len([test_y[i] for i in range(len(test_y)) if test_category[i] != 'Religion'])
    nonreligion_accuracy = max(nonreligion_accuracy, 1 - nonreligion_accuracy)
    print('non-religious performance', nonreligion_accuracy)
    print(args.model)
    total_accuracy = accuracy_score(test_y, pred_y)
    print('total performance', total_accuracy)
    religion_accuracy = accuracy_score([test_y[i] for i in range(len(test_y)) if test_category[i] == 'Religion'],
                                       [pred_y[i] for i in range(len(pred_y)) if test_category[i] == 'Religion'])
    print('religious performance', religion_accuracy)
    nonreligion_accuracy = accuracy_score([test_y[i] for i in range(len(test_y)) if test_category[i] != 'Religion'],
                                          [pred_y[i] for i in range(len(pred_y)) if test_category[i] != 'Religion'])
    print('non-religious performance', nonreligion_accuracy)
    #print([test_id[i] for i in range(len(test_id)) if test_y[i] == pred_y[i]])

    ###write out prediction results
    open(args.outfile, 'w').close()
    file = open(args.outfile, "w")
    pred_y = ['Pro' if p == 1 else 'Con' for p in pred_y]
    for debate in pred_y:
        file.write("%s\n" % debate)
    file.close()

