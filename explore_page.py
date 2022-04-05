import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df = pd.read_pickle("Models/saved_df")
X = pd.read_pickle("Models/X.pkl")
y = pd.read_pickle("Models/y.pkl")


def feature_selection():
    st.subheader("1. Univariate Selection")
    st.write("Importance of parameter based on their score."
             "Statistical tests can be used to select those features that have the strongest relationship with the output variable.")

    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Attribute Name', 'Score']  # naming the dataframe columns
    st.write(featureScores.nlargest(5, 'Score'))  # print 10 best features

    st.write("Here, rating has the strongest relationship with the output variable")


def feature_importance():
    st.subheader("2. Feature Importance")
    st.markdown("""
  You can get the feature importance of each feature of your dataset by using the feature importance property of the model.  
  Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.  
  Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.
  """)

    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(X, y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    fig = px.bar(feat_importances, orientation='h')
    st.write(fig)
    st.write("Visual representation of how strongly is output variable dependent on different input variables")


def correlation_matrix():
    st.subheader("3.Correlation Matrix with Heatmap")
    st.write("Correlation states how the features are related to each other or the target variable.")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)


def show_explore_page():
    st.title("""  Explore page """)
    feature_selection()
    feature_importance()
    correlation_matrix()

    st.subheader("4. Information for selected attributes")

    # checking is user has cliked the predict button
    if "flag" not in st.session_state:
        st.markdown(f"""
        #### <span style="color:#faa"> Please select inputs in predict page for further insights </span>
        """, unsafe_allow_html=True)
    else:
        # reding the data set
        df = pd.read_csv("MyCall_Data_September_2019_cleaning.csv")
        df.drop(['Latitude', 'Longitude'], axis='columns', inplace=True)
        df.rename(columns={
            'In Out Travelling': 'InOut', 'Network Type': 'NetworkType', 'Call Drop Category': 'CallDropCategory',
            'State Name': 'StateName'},
            inplace=True)

        # df = pd.read_pickle("Models/saved_df")
        old = df.copy()

        # setting the user selected attributes
        operator = st.session_state.operator
        inout = st.session_state.inout
        networktype = st.session_state.network
        statename = st.session_state.state
        ratings = st.session_state.rating

        # displaying the results
        d = old.groupby(['StateName']).size()
        d = d.reset_index()
        d1 = d[d['StateName'] == statename][0].to_string(index=False)

        d = old.groupby(['Operator']).size()
        d = d.reset_index()
        d2 = d[d['Operator'] == operator][0].to_string(index=False)

        d = old.groupby(['NetworkType']).size()
        d = d.reset_index()
        d3 = d[d['NetworkType'] == networktype][0].to_string(index=False)

        ans1 = str(old[
                       (old['StateName'] == statename) &
                       (old['Operator'] == operator) &
                       (old['NetworkType'] == networktype) &
                       (old['InOut'] == inout)
                       ].shape[0])

        st.markdown(f"""
      - In your state <span style="color:#8ef">{statename}</span> {str(d1)} calls were made  
      - For your operator <span style="color:#8ef">({operator}) </span> , {str(d1)} many calls were made  
      - For your network type <span style="color:#8ef">({(networktype)}) </span>  {str(d3)} calls were done          
      - In your state <span style="color:#fea">({statename}) </span>  with the operator <span style="color:#fea">({operator}) </span> ,while the connection being <span style="color:#fea">({inout}) </span> , there are a total of <span style="color:#8ef">{ans1} </span>  calls.             
        """, unsafe_allow_html=True)

        d = old.groupby(['Operator', 'CallDropCategory']).size()
        d = d.reset_index()
        d = d.rename(columns={0: 'Calls Dropped'})
        d = d[
            d['Operator'] == operator
            ]
        st.write("Category wise calls for your operator")
        st.write(d)

        st.subheader("5. Information about dataset")

        st.write("Calls made from each state")
        d = df.groupby(['StateName']).size()
        d = d.reset_index()
        d = d.rename(columns={0: 'Calls Dropped'})
        st.write(d)

        st.write("Calls from each operator")
        d = df.groupby(['Operator']).size()
        d = d.reset_index()
        d = d.rename(columns={0: 'Calls Dropped'})
        st.write(d)

        st.write("Calls from each network type")
        d = df.groupby(['NetworkType']).size()
        d = d.reset_index()
        d = d.rename(columns={0: 'Calls Dropped'})
        st.write(d)

        st.write("Statewise calls made")
        st.bar_chart(df.groupby(['StateName']).size())

        st.write("Average Ratind by Network Type, Separated by Operator")

        fig = plt.figure(figsize=(8, 5))
        sns.pointplot(x='NetworkType', y='Rating', data=df, hue='Operator')
        st.pyplot(fig)
