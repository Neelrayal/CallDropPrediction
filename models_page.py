from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn import metrics

indexx = ['Satisfactory', 'Poor Voice Quality', 'Call Dropped']


def fun1(y_test, y_pred):
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    output = pd.DataFrame(report).transpose()
    output = output[['precision', 'recall', 'f1-score']]
    output = output.head(3)
    output.index = indexx
    st.write(output.head(3))

    st.write(" #### Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)
    return metrics.accuracy_score(y_test, y_pred) * 100


def filename_replace(file_path):
    file_path = str(file_path)
    if "decisionTree" in file_path:
        return 'Decision Tree'
    elif "gaussNb" in file_path:
        return "Gauss NB"
    elif "knn" in file_path:
        return "KNN"
    elif "multiNb" in file_path:
        return "Multinomial NB"
    elif "randomForest" in file_path:
        return "Random Forest"
    elif "svm" in file_path:
        return "SVM"


def show_models_page():
    st.title(" Models page ")
    X = pd.read_pickle('Models/X.pkl')
    y = pd.read_pickle('Models/y.pkl')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    df = pd.DataFrame(columns=['Name', 'Accuracy'])
    dirs = Path("./Models").iterdir()
    for file_path in dirs:
        if "_model" in str(file_path):
            print(file_path)
            model = pd.read_pickle(file_path)
            model.fit(X_train.values, y_train)
            st.subheader(filename_replace(file_path))
            y_pred = model.predict(X_test)
            acc = fun1(y_test, y_pred)
            df.loc[len(df.index)] = [filename_replace(file_path), acc]

    st.subheader("Algorithm vs their accuracy")
    st.write(df)
