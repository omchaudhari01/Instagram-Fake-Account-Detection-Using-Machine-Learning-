import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def import_dataset(filePath):
    data = pd.read_csv(filePath)
    # Data Processing/ Data Munging
    # Encoding all the ordinal columns and creating a dummy variable for them to see if there are any effects on Performance Rating
    return data


def split_dataset(data, testSize):
    print(testSize / 100)
    # Here we have selected only the important columns
    X = data.drop('is_fake', axis=1)
    y = data.is_fake
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize / 100, random_state=42)
    print('Dataset Splited Successfully')
    print(len(X_train))
    print(len(X_test))
    return X_train, X_test, y_train, y_test


def view_data_correlation(data):
    plt.rcParams['figure.figsize'] = (30, 30)
    plt.matshow(data.corr())
    plt.yticks(np.arange(data.shape[1]), data.columns)
    plt.xticks(np.arange(data.shape[1]), data.columns)
    plt.colorbar()
    plt.show()


def train_svm(X_train, y_train):
    print('Training Algorithm .... !!!!!')
    svm_model = SVC(kernel='rbf', C=100, random_state=10).fit(X_train, y_train)
    print('Model Trained Successfully')
    return svm_model


def predict_svm(svm_model, X_test, y_test):
    print('Making Prediction')
    y_predict_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_predict_svm)
    print(accuracy_score(y_test, y_predict_svm))
    print(classification_report(y_test, y_predict_svm))
    print(y_predict_svm)
    return y_predict_svm, accuracy_svm


def train_log_reg(X_train, y_train):
    print('Training Algorithm .... !!!!!')
    rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_clf.fit(X_train, y_train)
    return rf_clf


def predict_log_reg(rf_clf, X_test, y_test):
    print('Making Prediction')
    y_predict_log = rf_clf.predict(X_test)
    accuracy_log = accuracy_score(y_test, y_predict_log)
    print(accuracy_score(y_test, y_predict_log))
    print(classification_report(y_test, y_predict_log))
    print(y_predict_log)
    return y_predict_log, accuracy_log


def train_ann(X_train, y_train):
    print('Training Algorithm .... !!!!!')
    model_ann = MLPClassifier(hidden_layer_sizes=(100, 100, 100), batch_size=100, learning_rate_init=0.01,
                              max_iter=2000, random_state=10)
    model_ann.fit(X_train, y_train)
    return model_ann


def predict_ann(model_ann, X_test, y_test):
    print('Making Prediction')
    y_predict_ann = model_ann.predict(X_test)
    accuracy_ann = accuracy_score(y_test, y_predict_ann)
    print(accuracy_score(y_test, y_predict_ann))
    print(classification_report(y_test, y_predict_ann))
    print(y_predict_ann)
    return y_predict_ann, accuracy_ann


def save_model(model, filename):
    with open(filename.name, 'wb') as file:
        pickle.dump(model, file)


# live_pred_dataset = pd.read_excel("dataset\live_prediction_dataset.xlsx")
# y = live_pred_dataset.PerformanceRating
# X = live_pred_dataset.iloc[:,[0,1,2,3,4,5,6,7,8]]
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)

## Load Log Reg Model
with open('model\\rf_clf.pkl', 'rb') as file:
    ran_for_model = pickle.load(file)

## Load SVM model
with open('model\svm_clf.pkl', 'rb') as file:
    svm_model = pickle.load(file)

## Load ANN model
with open('model\\ann_clf.pkl', 'rb') as file:
    ann_model = pickle.load(file)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
import numpy as np


def live_prediction(usr_data_list):
    rf_result = ran_for_model.predict(usr_data_list)
    rf_result_pro = ran_for_model.predict_proba(usr_data_list)
    rf_pred_perc = round(rf_result_pro[0][rf_result[0]] * 100, 2)
    print('RF : ', rf_result, rf_pred_perc, '%')

    svm_result = svm_model.predict(usr_data_list)
    svm_result_pro = svm_model.decision_function(usr_data_list)
    svm_result_pro = np.round(sigmoid(svm_result_pro), 3) # convert confidence scores to probabilities
    svm_pred_perc = round(svm_result_pro[0] * 100, 2)
    print('SVM : ', svm_result, svm_pred_perc, '%')

    ann_result = ann_model.predict(usr_data_list)
    ann_result_pro = ann_model.predict_proba(usr_data_list)
    ann_pred_perc = round(ann_result_pro[0][ann_result[0]] * 100, 2)
    print('ANN : ', ann_result, ann_pred_perc, '%')

    return rf_result, svm_result, ann_result


import numpy as np
from keras.models import load_model
insta_gru_model = load_model('model/insta_gru_best_model.h5')
insta_lstm_model = load_model('model/insta_lstm_best_model.h5')
insta_hybrid_model = load_model('model/insta_hybrid_best_model.h5')

def live_prediction_dl(usr_data_list):
    insta_gru_model_result = insta_gru_model.predict(usr_data_list)
    insta_gru_model_result = (insta_gru_model_result > 0.45)
    gru_result = 1
    if insta_gru_model_result[0][0]:gru_result=0
    print('insta_gru_model : ', gru_result)

    insta_lstm_model_result = insta_lstm_model.predict(usr_data_list)
    insta_lstm_model_result = (insta_lstm_model_result > 0.45)
    lstm_result = 1
    if insta_lstm_model_result[0][0]:lstm_result=0
    print('insta_lstm_model : ', lstm_result)

    input_data =  usr_data_list
    input_data_array = np.array(input_data)
    input_data_reshaped = input_data_array.reshape(1, -1)  # Reshape to (1, num_features)
    input_data_list = [input_data_reshaped, input_data_reshaped, input_data_reshaped]
    
    insta_hybrid_model_result = insta_hybrid_model.predict(input_data_list)
    insta_hybrid_model_result = (insta_hybrid_model_result > 0.45)
    hybrid_result = 1
    if insta_hybrid_model_result[0][0]:hybrid_result=0
    print('insta_gru_model : ', hybrid_result)


    return gru_result,lstm_result,hybrid_result


