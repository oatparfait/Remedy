import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from tkinter import *
from tkinter import messagebox
import sys 
import urllib
import urllib.request


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

# Note that earlier I mentioned that we have weighate against each symptom
# So we will simply perform an encoding operation here against each symptom

vals = training_set.values
symptoms = symptom_set['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = symptom_set[symptom_set['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)

# Weightage of these three aren't available in our dataset-2 hence as of now we are ignoring
d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination',0)
training_set = d.replace('foul_smell_of urine',0)

training_set.head()

# Now lets have a look at the different symptoms, we will need this list for option inputs in front-end
#list of symptoms 


# Can be referenced in front end web dev
training_set['Disease'].unique()

data = training_set.iloc[:,1:].values
data

# Y in prediction in terms of (X,Y)
labels = training_set['Disease'].values
labels

# After, we will train test split from the dataset 
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# The algorithm chosen is a support vector classifier machine.  They are a set of supervised machine learning methods.
model = SVC() # An instance of that model class

# Start training the model now
model.fit(x_train, y_train)

#x_test


# Predicting using test data ::
preds = model.predict(x_test)


# Model Metrics (Accuracy and others) ::
conf_mat = confusion_matrix(y_test, preds)
training_set_cm = pd.DataFrame(conf_mat, index=training_set['Disease'].unique(), columns=training_set['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
#sns.heatmap(training_set_cm)

# Dump the data and save in to ".sav" file for use in ML based front-end applications (Optional)
#import pickle
#pickle.dump(pred_model,open("svc_ml_model.sav", "wb"))