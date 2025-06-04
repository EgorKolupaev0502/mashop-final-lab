import csv
from datetime import datetime
from functools import partial
from itertools import product

import numpy as np
from keras._tf_keras import keras
from navec import Navec

from scipy.stats import gmean
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from keras import Sequential
from keras.src.layers import Embedding
import tensorflow as tf

from utils.data_loader import load_file, load_files
from utils.preprocessing import preprocess_sentence

sentences, aspects_marks = load_files(
                                      "data/aspects-marked-part-2-elections-1.csv",
                                      "data/aspects-marked-part-2-elections-2.csv",
                                      "data/aspects-marked-part-2-elections-3.csv"
                                      )

# lemmatize = [True, False]
# stem = [True, False]
# min_word_len = [0, 3]
lemmatize = [True]
stem = [False]
min_word_len = [0]
LSTM_SIZE = 1024
coef_dropout = 0.5
lstm = keras.layers.LSTM(LSTM_SIZE, dropout=coef_dropout, recurrent_dropout=coef_dropout)
def apply_navec(sentence, model_):
    unk = model['<unk>']
    words_matrix = np.array([model_.get(word, unk) for word in sentence])
    # word = words_matrix.mean(axis=0)
    # return word
    # print(word)
    words_vectors = []
    words_vectors.append(words_matrix)
    words_vectors = np.array(words_vectors)
    output = lstm(words_vectors)
    words_vectors_lstm = np.array(output[0])
    return words_vectors_lstm


models = [
          'models/navec_hudlit_v1_12B_500K_300d_100q.tar',
          'models/navec_news_v1_1B_250K_300d_100q.tar'
          ]

classifiers_builders = [
    lambda: RidgeClassifier(class_weight="balanced"),
    lambda: LogisticRegression(class_weight="balanced"),
    lambda: SVC(class_weight="balanced", random_state=42),
    lambda: LinearSVC(class_weight="balanced", random_state=42),
    lambda: SGDClassifier(class_weight="balanced", random_state=42),
    lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
    lambda: RandomForestClassifier(class_weight="balanced", random_state=42, max_depth=3),
    lambda : LogisticRegressionCV(class_weight="balanced")
]

best_f1, best_prec, best_rec = 0, 0, 0

results, results_header = [], ["model", "lemm", "stem", "min_word_len", "classifier", "avg_prec", "avg_rec", "avg_f1"]
result_console = []

dict_sentence = {}

for model_path in models:
    model = Navec.load(model_path)
    for lemmatize_, stem_, min_word_len_, in product(lemmatize, stem, min_word_len):
        print(lemmatize_, stem_, min_word_len_)
        try:
            X = np.array([apply_navec(preprocess_sentence(sentence, lemmatize_, min_word_len_, stem_, True), model) for sentence in sentences])
        except KeyError:
            print(model_path, lemmatize_, stem_, min_word_len_, "failed")
            results.append([model_path, lemmatize_, stem_, min_word_len_, "", None, None, None])
            continue
        for classifier_ in classifiers_builders:
            classifier = classifier_()
            result_console.append(f"{model_path} {classifier.__class__} {lemmatize_} {stem_} {min_word_len_}\n")
            f1_scores, precisions, recalls = [], [], []
            for aspect in aspects_marks.columns:
                y = aspects_marks[aspect].values.ravel()
                if y.sum() < 50:
                    continue
                y_pred = np.array([0] * y.shape[0])
                y_train_pred = np.array([0] * y.shape[0])
                for train_ix, test_ix in StratifiedKFold(n_splits=5).split(X, y):
                    X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix], y[test_ix]
                    clf = classifier
                    clf.fit(X_train, y_train)
                    y_train_out = clf.predict(X_train)
                    y_test_out = clf.predict(X_test)
                    y_train_pred[train_ix] = y_train_out
                    y_pred[test_ix] = y_test_out
                prec, rec, f1, support = [_[0] for _ in precision_recall_fscore_support(y, y_pred, labels=[1])]
                train_f1 = f1_score(y, y_train_pred, labels=[1])
                f1_scores.append(f1)
                precisions.append(prec)
                recalls.append(rec)
                result_console.append(f"Упоминание аспекта '{aspect:30}' ({support:3}): "
                      f"precision {prec:.2f}, recall {rec:.2f}, F1 {f1:.2f} (при обучении {train_f1:.2f})\n")

            avg_f1 = gmean(f1_scores).ravel()[0]
            avg_prec = gmean(precisions).ravel()[0]
            avg_rec = gmean(recalls).ravel()[0]
            results.append([model_path, lemmatize_, stem_, min_word_len_, classifier.__class__, avg_prec, avg_rec, avg_f1])
            tmp = list(map(lambda x: round(x, 3), [avg_prec, avg_rec, avg_f1]))
            for tmp1 in tmp:
                result_console.append(str(tmp1) + " ")
            if avg_f1 > best_f1:
                best_f1, best_prec, best_rec = avg_f1, avg_prec, avg_rec
            result_console.append("\n\n")

tmp = list(map(lambda x: round(x, 3), [best_prec, best_rec, best_f1]))
for tmp1 in tmp:
    result_console.append(str(tmp1) + " ")

# with open(f"journal/GloVe(navec)-lstm-{LSTM_SIZE}-{coef_dropout}-result-{datetime.now().date().strftime('%d%m%Y')}.csv", 'w', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(results_header)
#     writer.writerows(results)
#
# file = open(f"journal/GloVe(navec)-lstm-{LSTM_SIZE}-{coef_dropout}-result-{datetime.now().date().strftime('%d%m%Y')}.txt", "w")
# for str in result_console:
#     file.write(str)
# file.close()

with open(f"journal/GloVe(navec)-result-{datetime.now().date().strftime('%d%m%Y')}.csv", 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(results_header)
    writer.writerows(results)

file = open(f"journal/GloVe(navec)-result-{datetime.now().date().strftime('%d%m%Y')}.txt", "w")
for str in result_console:
    file.write(str)
file.close()