from pycaret.classification import *
import pandas as pd
from sklearn.neural_network import MLPClassifier
import random
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

random.seed(42)


def prepare_dataset(dataset):
    # Read allData
    data = pd.read_csv(dataset, sep=",", header=0,index_col=0)
    print(data.shape)

    # # Create a dataframe of darknet addresses
    # darknet = data.loc[data['label'] == "darknet"]
    # # Create a dataframe of ransomware addresses
    # ransom = data.loc[data['label'] == "ransomware"]
    #
    # # Select unique dates of darknet dataframe
    # darknet_dates = darknet['date'].unique()
    # # Select unique dates of random dataframe
    # ransom_dates = ransom['date'].unique()

    # # Create a dataframe of white addresses
    # white = data.loc[data['label'] == 'white']
    #
    # # Select white address that are in the same days of darknet addresses
    # white_dark = white.loc[white['date'].isin(darknet_dates)]
    # # Select white address that are in the same days of ransom addresses
    # white_ransom = white.loc[white['date'].isin(ransom_dates)]
    #
    # whiteSampleSize = 100000
    # darknetSampleSize = 100000
    dataSize = data.shape[0]

    # # Select sampled white and darknet data
    # whiteWithDarkDatesSampled = white_dark.sample(whiteSampleSize)
    # whiteWithRansomDatesSampled = white_ransom.sample(whiteSampleSize)
    # darknetSampled = darknet.sample(darknetSampleSize)
    #
    # data = pd.concat([whiteWithRansomDatesSampled, darknetSampled, ransom])
    # print(data['label'].value_counts())

    # Selecting only orbit features for classification
    orbit = data.drop(['address', 'year', 'month', 'day', 'count', 'length', 'weight', 'neighbors', 'looped', 'income','date'], axis = 1)
    # Selecting only heist features for classification
    heist = data[['count', 'length', 'weight', 'neighbors', 'looped', 'income','label']]

    # cols = [i for i in data.columns if i not in ["label"]]
    # for col in cols:
    #     data[col] = pd.to_numeric(data[col])
    # print(data.head())

    # Return orbit or heist data
    return orbit


def read_data(dataset):
    print(dataset.head())
    cr_setup = setup(data=dataset, target='label', train_size=0.9)  # setup environment
    best_models = compare_models(n_select=1)  # return first best model by comparing baseline models (in built models)
    df_bmodel = pull()  # convert best_models to an iterable
    best_model = df_bmodel.index[0]  # select the index for best model
    df_bmodel.to_csv("C:/Users/poupa/chainletorbits/results/ransomware/best_models.csv")

    return cr_setup, best_model


def create_predict_model(best_model):
    # Create model using best_model
    model = create_model(best_model, round=3)  # this function trains and evaluates a model using cross validation set by fold
    tune = tune_model(model)  # tuning the model parameters
    prediction = predict_model(tune)  # prediction
    df_results = pull()  # this returns the score grid obtained from prediction
    df_results.to_csv("C:/Users/poupa/chainletorbits/results/ransomware/metrics.csv")
    # plot_model(tune, plot = 'AUC', save=True)
    plot_model(tune, plot='confusion_matrix', save=True)

    if len(df_results.index) > 1:
        accuracy = df_results.loc['Mean']['Accuracy']
        # std = df_results['Std']['Accuracy']
    else:
        df_results = df_results.set_index('Model')
        accuracy = df_results['Accuracy']

    return accuracy


def mlp_classifier(cr_setup, best_model, accuracy):
    mlp = MLPClassifier()  # initialize classifier
    model = create_model(mlp, round=3)  # this function trains and evaluates a model using cross validation set by fold
    tune = tune_model(model)  # tuning the model parameters
    prediction = predict_model(tune)  # prediction
    df_results = pull()  # this returns the score grid obtained from prediction
    df_results.to_csv("C:/Users/poupa/chainletorbits/results/ransomware/mlpmetrics.csv")
    # plot_model(tune, plot = 'AUC_MLP', save=True)

    if len(df_results.index) > 1:
        mlp_accuracy = df_results.loc['Mean']['Accuracy']
        # std = df_results['Std']['Accuracy']
    else:
        df_results = df_results.set_index('Model')
        mlp_accuracy = df_results['Accuracy']

    file.write(str(cr_setup) + "\t" + (best_model) + "\t" + str(accuracy) + "\t" + str(mlp_accuracy) + "\n")

    file.flush()

def auc(dataset):



def main():
    dataset = prepare_dataset("C:/Users/poupa/chainletorbits/data/allDataModified.csv")
    # values_to_drop = ['EXCHANGE', 'MINING', 'SERVICE','GAMBLING']
    # dataset = dataset[~dataset['label'].isin(values_to_drop)]
    cr_setup, best_model = read_data(dataset)
    accuracy = create_predict_model(best_model)
    mlp_classifier(cr_setup, best_model, accuracy)


if __name__ == "__main__":
    outputfile = "C:/Users/poupa/chainletorbits/results/ransomware/" + "accuracy.csv"
    file = open(outputfile, 'w')
    main()

    file.close()
