## This Notebook has been released under the Apache 2.0 open source license.
import pandas as pd
import numpy as np
from tkinter import messagebox
import sys 
import urllib
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from tkinter import *


# Reading files into program
training_set = pd.read_csv(r"C:\Users\18457\PROGRAMMING\POWERINGSTEM-COMP-12.30.22\powerstem\dataset.csv")
symptom_set = pd.read_csv(r"C:\Users\18457\PROGRAMMING\POWERINGSTEM-COMP-12.30.22\powerstem\Symptom-severity.csv")


# Next I am cl eaning the data since parts of the set has a log of null values.
cols = training_set.columns
data = training_set[cols].values.flatten()
s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(training_set.shape)
training_set = pd.DataFrame(s, columns=training_set.columns)
training_set = training_set.fillna(0)
training_set.tail()

# Encoding to make it easier for machines to understand

vals = training_set.values
symptoms = symptom_set['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = symptom_set[symptom_set['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)

d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination',0)
training_set = d.replace('foul_smell_of urine',0)

training_set.head()

# Can be referenced in front end web dev
training_set['Disease'].unique()

data = training_set.iloc[:,1:].values
data

# Prediction 
labels = training_set['Disease'].values
labels

# Train test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = SVC() 

# Start training the model 
model.fit(x_train, y_train)

#x_test


# Predicting 
preds = model.predict(x_test)



# We can save the data into a file for it to be used in ML based front-end application
#import pickle
#pickle.dump(pred_model,open("svc_ml_model.sav", "wb"))
