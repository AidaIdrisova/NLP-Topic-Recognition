from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from Tops_functions import *
import pickle
import time

'''
The main aim of the program is to classify into general categories the dataset with conversations labelled by more 
specialised topics. 
The code implies the classification task trained on the train.csv dataset, tested on valid.csv dataset and used for 
bridged_10k.csv. Last dataset differs from the previous two containing the conversations instead of normal sentences, so 
all the phrases were classified and major category was chosen to the specialised topics. 
As training data appears to be highly unbalanced, several topics were combined('Emotional abuse', 'Suicide and SelfHarm'
 => 'Emotion', 'Sexual abuse' => 'Sex', 'Physical abuse' => 'Health', 'Terrorism / extremism' => 'Politics'), also the 
overloaded categories were limited to 5000 words each and for the categories containing less samples, the data was 
increased to this number of samples by smuto algotithm implemented on vectorized data. Glove vectorization was used on 
the tokenized words averaging for the sentences with removing stop_words. 
The dimensionality reduction of word vectors was implemented by the NMF.
For the classification the Support Vector Classification was used and the topics that did not get the probabilities more 
than 0.1 were moved to 'Others'. 
'''



def main(n_comp, n_samples_per_cat, Threshold_prob):
    seconds = time.time()

    file_test = "bridged_10k.csv"
    file_train = "train.csv"
    file_valid = "valid.csv"

    load_data(file_test, file_train, file_valid, n_samples_per_cat)

    with open('Simplified_smuto.pkl', 'rb') as fl:
        tt = pickle.load(fl)
    X_train, Y_train, X_test, Y_test, X_valid, Y_valid, Training_map, Validation_map = tt

    X_train, Y_train, X_valid, Y_valid, Training_map, Validation_map = cleaning_data_from_topics(X_train, Y_train, X_valid, Y_valid, Training_map, Validation_map, n_samples_per_cat)


    # Validation = Part of training !!!
    # Validation_map = Training_map
    # X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, random_state=42)


    # Vectorization
    model = load_Glove_Model()
    X_train_vect, X_train, Y_train = vect(X_train, Y_train, model)
    X_test_vect, X_test, Y_test = vect(X_test, Y_test, model)
    X_valid_vect, X_valid, Y_valid = vect(X_valid, Y_valid, model)

    # Making vectors positive
    min_val = 3.1
    X_train_vect = [x + min_val for x in X_train_vect]
    X_test_vect = [x + min_val for x in X_test_vect]
    X_valid_vect = [x + min_val for x in X_valid_vect]
    print('Vectorization is done')

    # # SMOTE
    # sm = SMOTE(random_state=42)
    # X_train_vect, Y_train = sm.fit_resample(X_train_vect, Y_train)

    ros = RandomOverSampler(random_state=42)
    X_train_vect, Y_train = ros.fit_resample(X_train_vect, Y_train)




    unique, counts = np.unique(Y_train, return_counts=True)
    print('Train data distribution after Smote:{}'.format(dict(zip(np.array(unique), np.array(counts)))))


    unique, counts = np.unique(Y_valid, return_counts=True)
    print('Valid data distribution after Smote:{}'.format(dict(zip(np.array(unique), np.array(counts)))))

    # Dimensionality reduction using NMF
    X_train_vect,  X_test_vect, X_valid_vect = dimensionality_reduction(X_train_vect,  X_test_vect, X_valid_vect,
                                                                     Y_train, n_comp)
    print('Dimensionality reduction is done')

    # Classification using SVC
    Y_pred, Y_train_pred, Y_valid_pred, Y_pred_prob, Y_train_pred_prob, Y_valid_pred_prob = classification(X_train_vect,
                                                                                                            X_test_vect, X_valid_vect, Y_train, Threshold_prob)

    data_total = [X_train, Y_train, X_test, Y_test, X_valid, Y_valid, Y_pred, Y_train_pred,
                  Y_valid_pred, Y_pred_prob, Y_train_pred_prob, Y_valid_pred_prob, Training_map, Validation_map]

    with open('Estimated.pkl', 'wb') as fs:
        pickle.dump(data_total, fs)

    with open('Estimated.pkl', 'rb') as fl:
        tt = pickle.load(fl)
    X_train, Y_train, X_test, Y_test, X_valid, Y_valid, Y_pred, Y_train_pred, Y_valid_pred,\
    Y_pred_prob, Y_train_pred_prob, Y_valid_pred_prob, Training_map, Validation_map = tt

    # print(np.unique(Y_train))
    # Diction = keys_output(Y_test, Y_pred, Y_pred_prob, Threshold=0.026)
    # final_excel(Diction)
    acc_valid, acc_train = efficiency_estimation(Y_valid_pred, Y_train_pred, X_valid, X_train, Training_map, Validation_map)

    unique, counts = np.unique(Y_valid, return_counts=True)
    print('Validation data distribution:{}'.format(dict(zip(np.array(unique), np.array(counts)))))

    X= "i" * len(Y_valid_pred)
    X, Y_valid_pred = flattening(X, Y_valid_pred)
    unique, counts = np.unique(Y_valid_pred, return_counts=True)
    print('Predicted Validation data distribution:{}'.format(dict(zip(np.array(unique), np.array(counts)))))

    print('Predicted categories amount: ', len(Y_valid_pred))
    print('True categories amount: ', len(Y_valid))
    mlb = MultiLabelBinarizer()
    mlb = mlb.fit(Y_valid)
    Y_valid = mlb.transform(Y_valid)
    Y_valid_pred = mlb.transform(Y_valid_pred)

    # matrix = confusion_matrix(Y_valid.argmax(axis =1), Y_valid_pred.argmax(axis =1))
    # print('Validation confusion matrix:'.format(matrix))



    return acc_valid, acc_train, seconds




if __name__ == "__main__":
    main(200, 1000, 0.16)

