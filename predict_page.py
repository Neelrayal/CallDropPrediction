import streamlit as st
import pickle
import numpy as np
import pandas as pd


df = pd.read_csv("MyCall_Data_September_2019_cleaning.csv")
df.drop(['Latitude', 'Longitude'], axis='columns', inplace=True)
df.rename(
	columns={'In Out Travelling': 'InOut', 'Network Type': 'NetworkType', 'Call Drop Category': 'CallDropCategory',
			 'State Name': 'StateName'}, inplace=True)
obj = {
	'Operator': {'RJio': 1, 'Airtel': 2, 'Idea': 3, 'Other': 4, 'Vodafone': 5, 'BSNL': 6, 'MTNL': 7},
	'InOut': {'Indoor': 1, 'Outdoor': 2, 'Travelling': 3},
	'NetworkType': {'4G': 1, '3G': 2, '2G': 3},
	'CallDropCategory': {'Satisfactory': 1, 'Poor Voice Quality': 2, 'Call Dropped': 3},
	'StateName': {}}

count = 1
for x in df['StateName'].unique():
	obj['StateName'][x] = count
	count += 1

dict1 = {1: 'Satisfactory', 2: 'Poor Voice Quality', 3: 'Call Dropped'}


def load_model(model):
	pick_read = open('Models/gaussNb_model.pkl', 'rb')
	if model == "Gauss NB":
		pass
	elif model == "Multinomial NB":
		pick_read = open('Models/multiNb_model.pkl', 'rb')
	elif model == "Decision Tree":
		pick_read = open('Models/decisionTree_model.pkl', 'rb')
	elif model == "KNN":
		pick_read = open('Models/knn_model.pkl.pkl', 'rb')
	elif model == "Neural Network":
		pick_read = open('Models/MLP_model.pkl', 'rb')
	elif model == "Random Forest":
		pick_read = open('Models/randomForest_model.pkl', 'rb')
	elif model == "SVM":
		pick_read = open('Models/svm_model.pkl', 'rb')
	data = pickle.load(pick_read)
	return data

def show_predict_page():
	st.title("Call Drop Analysis")
	st.write(
		"Analysed information based on the user inputs. These inputs will be passed to our pre trained model")

	operator = st.selectbox("Operator", list(obj['Operator'].keys()))
	inout = st.selectbox("Travelling type", list(obj['InOut'].keys()))
	network = st.selectbox("Network", list(obj['NetworkType'].keys()))
	state = st.selectbox("State", list(obj['StateName'].keys()))
	rating = st.slider("Rating", 1, 5, 1)
	classifier = st.selectbox("Classifier",
							  ("Gauss NB", "Multinomial NB", "Decision Tree", "KNN", "Neural Network", "Random Forest", "SVM"))
	btn = st.button('Precit Call Drop')
	if btn:
		model = load_model(classifier)
		temp = [[obj['Operator'][operator], obj['InOut'][inout], obj['NetworkType'][network], rating,
				 obj['StateName'][state]]]
		ans = model.predict(temp)[0]
		st.markdown(f"""
		    ### Predicted Category:  <span style="color:#8ef">{dict1[ans]} </span>
		  """, unsafe_allow_html=True)

		st.session_state.flag = True
		st.session_state.operator = operator
		st.session_state.inout = inout
		st.session_state.network = network
		st.session_state.state = state
		st.session_state.rating = rating

