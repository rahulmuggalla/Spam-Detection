#importing libraries 
import pandas as pd #for data analysis
import numpy as np #2 perform mathematical operations on arrays

from sklearn.feature_extraction.text import CountVectorizer #breaking down a sentence/paragraph/any text in2 words
from sklearn.model_selection import train_test_split #used 2 separate data as training & testing data
from sklearn.naive_bayes import MultinomialNB #counts words in text

#load the dataset 2 pandas data frame for manupulating the data
raw_data = pd.read_csv("D:\Projects\ML Projects\Deploy ML Project\Spam Mail Detection\spamham.csv", encoding = 'latin-1')

#now v hv 2 replace null values with null string otherwise it will show errors
#v will store this in variable claaed "mail_data"
mail_data = raw_data.where((pd.notnull(raw_data)), '')

#From this dataset, V1 & V2 are the only features we need to train a machine learning model for spam detection, 
#so let’s select these two columns as the new dataset
data = mail_data[["v1", "v2"]]

#now v need 2 separate te data as text & labels
x = np.array(data["v2"]) #x --> Text data
y = np.array(data["v1"]) #y --> Label data

#loading the CountVectorizer in2 the variable "cv"
cv = CountVectorizer()

#training the CountVectorizer wyt the text(v1) data
X = cv.fit_transform(x)

#now v need 2 split the dataset in2 training data & testing data
#train_size represents how how much % data u want for training samples
#test_size represents how much % data u want for testing 
#random_state splits the data in the specific way, u can put any variable u want
#if u want 2 split the data in the same way u did before v need 2 specify the same random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#training the model with training data
#data will b itterated around the hyperplane until gud plot is made
#training the MultinimialNB wyt the training data
#loading the MultinomialNB in2 the variable "model"
clf = MultinomialNB().fit(X_train, y_train)


#Web App

import streamlit as st #to create a web app

st.title("Spam Detection System") #title of the web app

def spamdetection():

    user = st.text_area("Enter any Message or Email : ") #taking input from the user

    if len(user) < 1:
        st.write("  ")

    else:

        #Predictive model
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)

spamdetection()