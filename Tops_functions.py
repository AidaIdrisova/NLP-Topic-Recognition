import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from nltk.corpus import wordnet as wn
from sklearn.svm import SVC
from sklearn.utils import resample
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import pickle


stop_words = set(stopwords.words('english'))

def load_data(file_test, file_train, file_valid, n_samples_per_cat):
    test = pd.read_csv(file_test)
    train = pd.read_csv(file_train, encoding="ISO-8859-1")
    valid = pd.read_csv(file_valid)

    # # ? test['Total'] = [x.str.cat() for x in test]
    test['Total'] = '' * (len(test))
    for j in range(len(test)):
        row = test.iloc[j]
        test.Total[j] = row.str.cat()

    test = test[['Total']]
    test = test.Total.str.split('|', expand=True)
    test.columns = ['Categories', '1', '2', '3', '4', '5', '6', '7','8']
    test.Categories = [x.rstrip() for x in test.Categories]

    X_test = []
    Y_test = []
    col = test.columns
    for i in range(len(test)):
        row = test.iloc[i]
        for k in range(1, len(test.columns)):
            if str(row[col[k]]) != 'nan' and str(row[col[0]]) != 'nan':
                sent = row[col[k]]
                X_test.append(sent.rstrip())
                cat = row[col[0]]
                Y_test.append(cat.rstrip())

    Validation_map ={}
    X_valid = []
    Y_valid = []
    col = valid.columns
    for i in valid.iterrows():
        i = i[1]
        if str(i[col[0]]) != 'nan':
            categories = i[col[1]]
            categories = categories.split(',')
            categories = [x.rstrip() for x in categories if x != 'Categories']
            Y_valid.append(categories)
            X_valid.append(i[col[0]])
            Validation_map[i[col[0]]] = categories

    X_valid, Y_valid = flattening(X_valid, Y_valid)

    tt = X_test, Y_test, X_valid,Y_valid
    with open('Test.pkl', 'wb') as fs:
        pickle.dump(tt, fs)

    with open('Test.pkl', 'rb') as fl:
        tt = pickle.load(fl)
    X_test, Y_test, X_valid, Y_valid = tt
    print('')

    X_train_multiple = []
    Y_train_multiple = []
    X_train_single = []
    Y_train_single = []
    Training_map = {}

    col = train.columns
    for i in train.iterrows():
        i = i[1]
        if str(i[col[0]]) != 'nan':
            categories = i[col[1]]
            categories = categories.split(',')
            Training_map[i[col[0]]] = categories
            if len(categories) == 1:
                X_train_single.append(i[col[0]])
                Y_train_single.append(categories[0])
            else:
                X_train_multiple.append(i[col[0]])
                Y_train_multiple.append(categories)


    X_train_multiple, Y_train_multiple = flattening(X_train_multiple, Y_train_multiple)

    X_train, Y_train = training_data(X_train_single, Y_train_single, X_train_multiple, Y_train_multiple, n_samples_per_cat)

    unique, counts = np.unique(np.array(Y_train), return_counts=True)
    print('Train data distribution:{}'.format(dict(zip(np.array(unique), np.array(counts)))))

    data_total = [X_train, Y_train, X_test, Y_test, X_valid, Y_valid, Training_map, Validation_map]

    with open('Simplified_smuto.pkl', 'wb') as fs:
        pickle.dump(data_total, fs)


def flattening(X, Y):
    X_t = []
    Y_t = []
    for i, x in enumerate(Y):
        for k in x:
            Y_t.append(k)
            X_t.append(X[i])
    return X_t, Y_t


def training_data(X_train_single, Y_train_single, X_train_multiple, Y_train_multiple, n_samples_per_cat):
    X_train = []
    Y_train = []

    unique_single, indexes_single, counts_single = np.unique(Y_train_single, return_counts=True, return_inverse=True)
    for i in range(len(unique_single)):
        add_number = n_samples_per_cat - counts_single[i]

        X_t_single = [X_train_single[k] for k in range(len(Y_train_single)) if Y_train_single[k] == unique_single[i]]
        Y_t_single = [Y_train_single[k] for k in range(len(Y_train_single)) if Y_train_single[k] == unique_single[i]]
        X_t_multiple = [X_train_multiple[k] for k in range(len(Y_train_multiple)) if
                        Y_train_multiple[k] == unique_single[i]]
        Y_t_multiple = [Y_train_multiple[k] for k in range(len(Y_train_multiple)) if
                        Y_train_multiple[k] == unique_single[i]]

        X_cat = []
        Y_cat = []
        if add_number <= 0:
            X_cat, Y_cat = resample(X_t_single, Y_t_single, replace=False, n_samples=n_samples_per_cat, random_state=22)
        elif add_number < n_samples_per_cat and add_number > 0:
            X_train.extend(X_t_single)
            Y_train.extend(Y_t_single)
            if add_number < len(X_t_multiple):
                X_cat, Y_cat = resample(X_t_multiple, Y_t_multiple,
                                        replace=False, n_samples=add_number, random_state=22)
            else:
                X_cat, Y_cat = (X_t_multiple, Y_t_multiple)
        X_train.extend(X_cat)
        Y_train.extend(Y_cat)
    return X_train, Y_train


def cleaning_data_from_topics(x_train, y_train, x_valid, y_valid, training_map, validation_map, n_samples_per_cat):
    unique, index,  counts = np.unique(np.array(y_train), return_counts=True, return_inverse=True)
    cat_to_remove = unique[counts < n_samples_per_cat]
    print('Removed Categories: {}'.format(cat_to_remove))

    x_train = [x_train[i] for i in range(len(x_train)) if y_train[i] not in cat_to_remove]
    y_train = [y_train[i] for i in range(len(y_train)) if y_train[i] not in cat_to_remove]

    x_valid = [x_valid[i] for i in range(len(x_valid)) if y_valid[i] not in cat_to_remove]
    y_valid = [y_valid[i] for i in range(len(y_valid)) if y_valid[i] not in cat_to_remove]

    for cat in training_map:
        if set(training_map[cat]) & set(cat_to_remove) != []:
            if set(training_map[cat]) & set(cat_to_remove) == set(training_map[cat]):
                training_map[cat] = ''
            else:
                training_map[cat] = list(set(training_map[cat]) - set(cat_to_remove))

    for cat in validation_map:
        if set(validation_map[cat]) & set(cat_to_remove) != []:
            if set(validation_map[cat]) & set(cat_to_remove) == set(validation_map[cat]):
                validation_map[cat] = ''
            else:
                validation_map[cat] = list(set(validation_map[cat]) - set(cat_to_remove))

    return x_train, y_train, x_valid, y_valid, training_map, validation_map


def data_for_conversations(X_train, Y_train):
    Y_train = combining_categories(Y_train)
    for n, i in enumerate(X_train):
        if i == 'Others' or i == 'Existential':
            del X_train[n]
            del Y_train[n]
    return X_train, Y_train


def combining_categories(data):
    for i, categories in enumerate(data):
        if type(categories == str):
            if categories == 'Physical abuse':
                data[i] = 'Health'
            if categories == 'Emotional abuse':
                data[i] = 'Emotion'
            if categories == 'Sexual abuse':
                data[i] = 'Sex'
            if categories == 'Suicide and self harm':
                data[i] = 'Emotion'
            if categories == 'Terrorism / extremism':
                data[i] = 'Politics'
        else:
            len_cat = len(categories)
            for k in range(len_cat):
                if categories[k] == 'Physical abuse':
                    categories[k] = 'Health'
                if categories[k] == 'Emotional abuse':
                    categories[k] = 'Emotion'
                if categories[k] == 'Sexual abuse':
                    categories[k] = 'Sex'
                if categories[k] == 'Suicide and self harm':
                    categories[k] = 'Emotion'
                if categories[k] == 'Terrorism / extremism':
                    categories[k] = 'Politics'
            data[i] = categories
    return data


def vect(x, y, model):
    x_rest = x
    x = [get_tokens(i) for i in x]
    vect = []
    ind = []
    for i in range(0, len(x)):
        tokens = x[i]
        if any(token in model.keys() for token in tokens):
            vect.append(np.mean([model[token] for token in tokens if token in model.keys()], axis=0))
        else:
            ind.append(i)
    y = [i for j, i in enumerate(y) if j not in ind]
    x_rest = [i for j, i in enumerate(x_rest) if j not in ind]
    return vect, x_rest, y


# def vect_Glove(x, y, model):
#     x = [get_tokens(i) for i in x]
#     vect = []
#     ind = []
#     for i in range(0, len(x)):
#         tokens = x[i]
#         if tokens != 'nan':
#             vect.append(np.mean([model[token] for token in tokens], axis=0))
#         else:
#             ind.append(i)
#     y = [i for j, i in enumerate(y) if j not in ind]
#     return vect, y


def get_tokens(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [token for token in tokens if (token not in stop_words and len(token) > 1)]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def load_Glove_Model():
    glove_file = 'glove.6B.300d.txt'
    f = open(glove_file, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def create_Glove_Model(data):
    corpus = Corpus()
    corpus.fit(data, window=10)
    glove = Glove(no_components=50, learning_rate=0.05)

    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')
    return glove.word_vectors


def dimensionality_reduction(x_train, x_test, x_valid, y_train, n_comp):
    # pca = TruncatedSVD(n_components=20)
    print(n_comp)
    pca = NMF(n_components=n_comp)
    # pca = LatentDirichletAllocation(n_components=200)
    # print(x_train)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    x_valid = pca.transform(x_valid)
    return x_train, x_test, x_valid


def classification(X_train, X_test, X_valid, Y_train, Threshold_prob):
    # classifier = GaussianNB()
    classifier = SVC(probability=True)
    # classifier = RandomForestClassifier()
    classifier.fit(X_train, Y_train)
    classes = classifier.classes_

    Y_pred_prob = classifier.predict_proba(X_test)
    Y_train_pred_prob = classifier.predict_proba(X_train)
    Y_valid_pred_prob = classifier.predict_proba(X_valid)

    Y_pred, Y_pred_prob = low_class_remove(Y_pred_prob, classes, Threshold=0)
    Y_train_pred, Y_train_pred_prob = low_class_remove(Y_train_pred_prob, classes, Threshold=Threshold_prob)
    Y_valid_pred, Y_valid_pred_prob = low_class_remove(Y_valid_pred_prob, classes, Threshold=Threshold_prob)

    return Y_pred, Y_train_pred, Y_valid_pred, Y_pred_prob, Y_train_pred_prob, Y_valid_pred_prob


def low_class_remove(y_prob, classes, Threshold):
    # classes = classes.tolist()
    y = [None] * len(y_prob)
    y_prob_classes = [None] * len(y_prob)

    for n, i in enumerate(y_prob):
        if all(i < Threshold):
            y[n] = ['Other']
            y_prob_classes[n] = [Threshold]
        else:
            y[n] = classes[i > Threshold].tolist()
            y_prob_classes[n] = i[i > Threshold]
            # if 'Other' in y[n] and len(y[n]) > 1:
            #     y[n] = del
    return y, y_prob_classes


def keys_output(Y_test, Y_pred, Y_pred_prob, Threshold):
    Diction = {}
    Final = pd.DataFrame([Y_test, Y_pred, Y_pred_prob])
    Final = np.transpose(Final)
    Final.columns = ['Categories', 'Classes', 'Probabilities']
    grouped = Final.groupby(['Categories'])
    for name, cat in grouped:
        clas = [st for row in cat.Classes for st in row]
        probs = [st for row in cat.Probabilities for st in row]
        number_of_sen = len(clas)
        unique, pos = np.unique(clas, return_inverse=True)
        probs = np.array(probs)
        # Probs = [sum(x) for i in range(len(unique)) for x in probs[pos == i]]
        Probs = np.zeros(len(unique))
        for i in range(len(unique)):
            Probs[i] = sum(probs[pos == i])/number_of_sen
        ind_max = Probs.argsort()
        Probs = Probs[ind_max]
        unique = unique[ind_max]
        if all(Probs < Threshold):
            Diction[name] = unique[Probs.argmax()]
        else:
            topics = unique[Probs > Threshold]
            if 'Other' in topics and len(topics) > 1:
                topics = topics[topics != 'Other']
            Diction[name] = topics
    with open('Topics_keys.csv', 'w') as output:
        writer = csv.writer(output)
        for key, value in Diction.items():
            writer.writerow([key, np.unique(value)])
    return Diction


def final_excel(Diction):
    with open('Test.pkl', 'rb') as fl:
        tt = pickle.load(fl)
    test = tt
    test['Topics'] = [Diction[x] for x in test['Categories']]
    test.to_csv('Topics.csv')


def efficiency_estimation(Y_valid_pred, Y_train_pred, X_valid, X_train, training_map, validation_map):
    n = 0
    total = 0
    for i in range(0, len(X_valid)):
        categories_true = validation_map[X_valid[i]]
        # print('True categories: ', categories_true,' Predicted:', Y_valid_pred[i], '\n')
        for k in Y_valid_pred[i]:
            if k in categories_true:
                n = n + 1
        total += len(categories_true)
    acc_valid = 100*n/total
    print('The accuracy on the valid text : {}%'.format(acc_valid))

    n = 0
    total = 0
    for i in range(0, len(X_train)):
        categories_true = training_map[X_train[i]]
        for k in Y_train_pred[i]:
            if k in categories_true:
                n = n + 1
        total += len(categories_true)
    acc_train = 100*n/total
    print('The accuracy on the train text : {}%'.format(acc_train))
    return acc_valid, acc_train


# test_new = test.groupby('Category', group_keys=False)
# for i in test_new.groups:
#     grouped = test_new.get_group(i)
#     grouped_list = grouped[col[1:15]].values.tolist()
#     grouped_list = list(itertools.chain.from_iterable(grouped_list))
#     grouped_list = [x for x in grouped_list if str(x) != 'nan' and word_tokenize(x) != stop_words]
#
# Rake().extract_keywords_from_text(text)
# ranked_phrases = Rake().get_ranked_phrases()

# def vectorize(X_train, X_train_acc, X_test, X_valid):
#     Total = X_train + X_test + X_valid
#     vectorizer = CountVectorizer(analyzer='word', max_df=0.95, min_df=2, stop_words='english', lowercase=True)
#     # vectorizer = TfidfVectorizer()
#     vectorizer.fit(Total)
#     X_test = vectorizer.transform(X_test)
#     X_train = vectorizer.transform(X_train)
#     X_valid = vectorizer.transform(X_valid)
#     X_train_acc = vectorizer.transform(X_train_acc)
#     return X_train, X_train_acc, X_test, X_valid

 # unique, indexes, counts = np.unique(Y_train, return_counts=True, return_inverse=True)
    # X_train_new = []
    # Y_train_new = []
    # n_samples_categ = 1000
    # for i in range(len(unique)):
    #     ind = (indexes == i)
    #     X_cat = np.array(X_train)[ind]
    #     Y_cat = np.array(Y_train)[ind]
    #     if sum(ind) < n_samples_categ:
    #         X_cat, Y_cat = resample(X_cat, Y_cat, replace=True, n_samples=n_samples_categ, random_state=22)
    #     else:
    #         X_cat, Y_cat = resample(X_cat, Y_cat, replace=False, n_samples=n_samples_categ, random_state=22)
    #     X_train_new.extend(X_cat)
    #     Y_train_new.extend(Y_cat)
    #
    # data_total = [X_train, Y_train, X_train_acc, Y_train_acc, X_test, Y_test, X_valid, Y_valid]
    #
    # with open('Simplified.pkl', 'wb') as fs:
    #     pickle.dump(data_total, fs)

    # X_train, Y_train = data_for_conversations(X_train, Y_train)

    # n_components = np.arange(12)
    # # n_components = np.arange(1)
    # Valid = []
    # Train = []
    # Time = []
    # n_components = [x*20 + 50 for x in n_components]
    # for i in n_components:
    #     acc_valid, acc_train, seconds = main(i)
    #     Valid.append(acc_valid)
    #     Train.append(acc_train)
    #     Time.append(seconds)
    #
    # with open('output.pkl', 'wb') as fs:
    #     pickle.dump([n_components, Valid, Train, Time], fs)