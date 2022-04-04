
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


global X_train, X_test, y_train, y_test

def already_built(X_train, X_test, y_train, y_test):

    from sklearn.naive_bayes import GaussianNB
    gaussNb = GaussianNB()
    gaussNb.fit(X_train.values, y_train.values)

    # predicting test set results
    y_pred = gaussNb.predict(X_test)

    # saving the model as pickel module
    with open('Models/gaussNb_model.pkl', 'wb') as file:
        pickle.dump(gaussNb, file)

    print("\nGaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.classification_report(y_test, y_pred))

    from sklearn.naive_bayes import MultinomialNB
    MultiNb = MultinomialNB()
    MultiNb.fit(X_train.values, y_train.values)

    # saving the model as pickel module
    with open('Models/multiNb_model.pkl', 'wb') as file:
        pickle.dump(MultiNb, file)

    y_pred = MultiNb.predict(X_test)
    print("\Multinomail Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.classification_report(y_test, y_pred))

    from sklearn import tree
    decisionTree = tree.DecisionTreeClassifier()

    decisionTree.fit(X_train.values, y_train.values)
    y_pred = decisionTree.predict(X_test)

    with open('Models/decisionTree_model.pkl', 'wb') as file:
        pickle.dump(decisionTree, file)
    print("\nDecision Tree(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.classification_report(y_test, y_pred))

    from sklearn.ensemble import RandomForestClassifier
    randomForest = RandomForestClassifier(n_estimators=25)

    randomForest.fit(X_train.values, y_train.values)
    y_pred = randomForest.predict(X_test)

    with open('Models/randomForest_model.pkl', 'wb') as file:
        pickle.dump(randomForest, file)
    print("\nRandomForest(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.classification_report(y_test, y_pred))

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train.values, y_train.values)
    y_pred = knn.predict(X_test)

    with open('Models/knn_model.pkl', 'wb') as file:
        pickle.dump(knn, file)
    print("\nKNN (in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.classification_report(y_test, y_pred))

    from sklearn.neural_network import MLPClassifier
    MLP = MLPClassifier()
    MLP.fit(X_train.values, y_train.values)
    y_pred = MLP.predict(X_test)

    with open('Models/MLP_model.pkl', 'wb') as file:
        pickle.dump(MLP, file)
    print("\nMLP (in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.classification_report(y_test, y_pred))

def feature_selection(X, y):
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features

def feature_importance(X,y):
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

    # Feature importance gives you a score for each feature of your data,
    # the higher the score more important or relevant is the feature towards your output variable.

def correelation_matrix(X,y, data):
    # get correlations of each features in dataset

    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")


def pre():
    df = pd.read_csv("MyCall_Data_September_2019_cleaning.csv")

    df.drop(['Latitude', 'Longitude'], axis='columns', inplace=True)

    df.rename(columns={
        'In Out Travelling': 'InOut', 'Network Type': 'NetworkType', 'Call Drop Category': 'CallDropCategory',
        'State Name': 'StateName'
    }, inplace=True)

    old = df.copy()

    obj = {
        'Operator': {'RJio': 1, 'Airtel': 2, 'Idea': 3, 'Other': 4, 'Vodafone': 5, 'BSNL': 6, 'MTNL': 7},
        'InOut': {'Indoor': 1, 'Outdoor': 2, 'Travelling': 3},
        'NetworkType': {'4G': 1, '3G': 2, '2G': 3},
        'CallDropCategory': {'Satisfactory': 1, 'Poor Voice Quality': 2, 'Call Dropped': 3},
        'StateName': {}
    }

    count = 1
    for x in df['StateName'].unique():
        obj['StateName'][x] = count
        count += 1

    # Do not run this code two times, otherwise the map function will convert all non-matched values to NaN

    df['Operator'] = df['Operator'].map(
        {'RJio': 1, 'Airtel': 2, 'Idea': 3, 'Other': 4, 'Vodafone': 5, 'BSNL': 6, 'MTNL': 7})

    df['InOut'] = df['InOut'].map({'Indoor': 1, 'Outdoor': 2, 'Travelling': 3})

    df['NetworkType'] = df['NetworkType'].map({'4G': 1, '3G': 2, '2G': 3})

    df['CallDropCategory'] = df['CallDropCategory'].map({'Satisfactory': 1, 'Poor Voice Quality': 2, 'Call Dropped': 3})

    df['StateName'] = df['StateName'].map(obj['StateName'])

    X = df.columns.tolist()
    X.remove('CallDropCategory')
    X = df[X]
    X.to_pickle('Models/X.pkl')

    y = df['CallDropCategory']
    y.to_pickle('Models/y.pkl')




def build_model(X_train, X_test, y_train, y_test):
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    from sklearn.neural_network import MLPClassifier
    MLP = MLPClassifier()
    MLP.fit(X_train.values,y_train.values)
    y_pred = MLP.predict(X_test)

    with open('Models/MLP_model.pkl', 'wb') as file:
        pickle.dump(MLP, file)
    print("\nMLP (in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    print(metrics.classification_report(y_test, y_pred))


def main():
    df = pd.read_pickle("Models/saved_df")
    X = pd.read_pickle("Models/X.pkl")
    y = pd.read_pickle("Models/y.pkl")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    build_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()



