import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn import metrics
import os


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
'''
def plot_metrics(metrics_list, model):
	st.set_option('deprecation.showPyplotGlobalUse', False)

	if 'Confusion Matrix' in metrics_list:
		st.subheader("Confusion Matrix")
		plot_confusion_matrix(model, X_test, y_test, display_labels=indexx)
		st.pyplot()

	if 'ROC Curve' in metrics_list:
		st.subheader("ROC Curve")
		plot_roc_curve(model, X_test, y_test)
		st.pyplot()

	if 'Precision-Recall Curve' in metrics_list:
		st.subheader("Precision-Recall Curve")
		plot_precision_recall_curve(model, X_test, y_test)
		st.pyplot()
'''

filename_replace = {
  "Models\decisionTree_model.pkl" : 'Decision Tree',
  "Models\gaussNb_model.pkl" : "Gauss NB",
  "Models\knn_model.pkl" : "KNN",
  "Models\MLP_model.pkl": "Neural Network",
  "Models\multiNb_model.pkl": "Multinomial NB",
  r"Models\randomForest_model.pkl": "Random Forest",
  "Models\svm_model.pkl": "SVM"
}


def show_models_page():
  st.title(" Models page ")
  X = pd.read_pickle('Models/X.pkl')
  y = pd.read_pickle('Models/y.pkl')
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

  directory = 'Models'

  df = pd.DataFrame(columns=['Name', 'Accuracy'])
  for filename in os.scandir(directory):
    if filename.is_file() and "_model.pkl" in filename.path:
      model = pd.read_pickle(filename.path)
      model.fit(X_train.values, y_train)
      st.subheader(filename_replace[filename.path])
      y_pred = model.predict(X_test)

      acc = fun1(y_test, y_pred)

      df.loc[len(df.index)] = [filename_replace[filename.path], acc]


  st.subheader("Algorithm vs their accuracy")
  st.write(df)


