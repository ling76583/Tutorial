import os
import argparse
from transformers import BertTokenizer, TFBertModel

from sklearn.ensemble import RandomForestClassifier
import numpy as np


import time
start_time = time.time()

LABELS = ['F', 'T']


def get_wic_subset(data_dir):
	wic = []
	split = data_dir.strip().split('/')[-1]
	with open(os.path.join(data_dir, '%s.data.txt' % split), 'r', encoding='utf-8') as datafile, \
		open(os.path.join(data_dir, '%s.gold.txt' % split), 'r', encoding='utf-8') as labelfile:
		for (data_line, label_line) in zip(datafile.readlines(), labelfile.readlines()):
			word, _, word_indices, sentence1, sentence2 = data_line.strip().split('\t')
			sentence1_word_index, sentence2_word_index = word_indices.split('-')
			label = LABELS.index(label_line.strip())
			wic.append({
				'word': word,
				'sentence1_word_index': int(sentence1_word_index),
				'sentence2_word_index': int(sentence2_word_index),
				'sentence1_words': sentence1.split(' '),
				'sentence2_words': sentence2.split(' '),
				'label': label
			})
	return wic


def embedding_data(wic, first_run, train_or_test):
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	model = TFBertModel.from_pretrained("bert-base-cased")

	if first_run:
		left_embd = []
		right_embd = []
		label = []
		sent1_index = []
		sent2_index = []
		for pair in wic:
			label.append(pair['label'])
			sent1_index.append(pair['sentence1_word_index'])
			sent2_index.append(pair['sentence2_word_index'])
			text = ' '.join(pair['sentence1_words'])
			left_embd.append(text)

			text = ' '.join(pair['sentence2_words'])
			right_embd.append(text)

		encoded_input = tokenizer(left_embd, return_tensors='tf', padding=True)
		output = model(encoded_input)
		left_embd = output[0].numpy()
		left_embd = [left_embd[i][sent1_index[i]] for i in range(len(left_embd))]
		left_embd = np.asarray(left_embd)
		np.savetxt(train_or_test + "left.csv", left_embd, delimiter=",")

		encoded_input = tokenizer(right_embd, return_tensors='tf', padding=True)
		output = model(encoded_input)
		right_embd = output[0].numpy()
		right_embd = [right_embd[i][sent2_index[i]] for i in range(len(right_embd))]
		right_embd = np.asarray(right_embd)
		np.savetxt(train_or_test + "right.csv", right_embd, delimiter=",")

		label = np.asarray(label)
		np.savetxt(train_or_test + "label.csv", label, delimiter=",")

	left_embd = np.genfromtxt(train_or_test + 'left.csv', delimiter=',')
	right_embd = np.genfromtxt(train_or_test + 'right.csv', delimiter=',')

	dataset = {}
	dataset['data'] = np.subtract(left_embd, right_embd)
	dataset['label'] = np.genfromtxt(train_or_test + 'label.csv', delimiter=',')
	print("--- %s seconds ---" % (time.time() - start_time))
	return dataset


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Train a classifier to recognize words in context (WiC).'
	)
	parser.add_argument(
		'--train-dir',
		dest='train_dir',
		required=True,
		help='The absolute path to the directory containing the WiC train files.'
	)
	parser.add_argument(
		'--eval-dir',
		dest='eval_dir',
		required=True,
		help='The absolute path to the directory containing the WiC eval files.'
	)
	# Write your predictions (F or T, separated by newlines) for each evaluation
	# example to out_file in the same order as you find them in eval_dir.  For example:
	# F
	# F
	# T
	# where each row is the prediction for the corresponding line in eval_dir.
	parser.add_argument(
		'--out-file',
		dest='out_file',
		required=True,
		help='The absolute path to the file where evaluation predictions will be written.'
	)
	args = parser.parse_args()

	train_wic = get_wic_subset(args.train_dir)
	eval_wic = get_wic_subset(args.eval_dir)

	FIRST_RUN = False
	train = embedding_data(train_wic, False, 'train')
	dev = embedding_data(eval_wic, False, 'test')

	clf = RandomForestClassifier(n_estimators=200, max_features='log2', max_depth=200, min_samples_leaf=5, random_state=0)
	clf.fit(train['data'], train['label'])
	predicted = clf.predict(dev['data'])
	print("Accuracy: ")
	print(len([i for i in range(len(dev['label'])) if dev['label'][i] == predicted[i]]) / len(dev['label']))
	print([(i, dev['label'][i]) for i in range(len(dev['label'])) if dev['label'][i] != predicted[i]])

	with open(args.out_file, 'w') as f:
		for element in predicted:
			if element == 1:
				f.write("T\n")
			else:
				f.write("F\n")





