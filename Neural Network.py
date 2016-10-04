
# coding: utf-8

### detect the fake profiles in online social networks using Neural Network

# In[1]:

import sys
import csv
import os
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sexmachine.detector as gender
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from  sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_curve, auc ,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
get_ipython().magic(u'matplotlib inline')
from pybrain.structure import SigmoidLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader


####### function for reading dataset from csv files

# In[2]:

def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("data/users.csv")
    fake_users = pd.read_csv("data/fusers.csv")
    # print genuine_users.columns
    # print genuine_users.describe()
    #print fake_users.describe()
    x=pd.concat([genuine_users,fake_users])   
    y=len(fake_users)*[0] + len(genuine_users)*[1]
    return x,y
    


####### function for predicting sex using name of person

# In[3]:

def predict_sex(name):
    sex_predictor = gender.Detector(unknown_value=u"unknown",case_sensitive=False)
    first_name= name.str.split(' ').str.get(0)
    sex= first_name.apply(sex_predictor.get_gender)
    sex_dict={'female': -2, 'mostly_female': -1,'unknown':0,'mostly_male':1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code


####### function for feature engineering

# In[4]:

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name : i for i, name in lang_list }             
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x.loc[:,'sex_code']=predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lang_code']
    x=x.loc[:,feature_columns_to_use]
    return x


####### function for plotting confusion matrix

# In[5]:

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names=['Fake','Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


####### function for plotting ROC curve

# In[6]:

def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    print "False Positive rate: ",false_positive_rate
    print "True Positive rate: ",true_positive_rate


    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


####### Function for training data using Neural Network

# In[7]:

def train(X,y):
    """ Trains and predicts dataset with a Neural Network classifier """

    ds = ClassificationDataSet( len(X.columns), 1,nb_classes=2)
    for k in xrange(len(X)): 
    	ds.addSample(X.iloc[k],np.array(y[k]))
    tstdata, trndata = ds.splitWithProportion( 0.20 )
    trndata._convertToOneOfMany( )
    tstdata._convertToOneOfMany( ) 
    input_size=len(X.columns)
    target_size=1
    hidden_size = 5   
    fnn=None
    if  os.path.isfile('fnn.xml'): 
    	fnn = NetworkReader.readFrom('fnn.xml') 
    else:
    	fnn = buildNetwork( trndata.indim, hidden_size , trndata.outdim, outclass=SoftmaxLayer )	
    trainer = BackpropTrainer( fnn, dataset=trndata,momentum=0.05, learningrate=0.1 , verbose=False, weightdecay=0.01)


    trainer.trainUntilConvergence(verbose = False, validationProportion = 0.15, maxEpochs = 100, continueEpochs = 10 )
    NetworkWriter.writeToFile(fnn, 'oliv.xml')
    predictions=trainer.testOnClassData (dataset=tstdata)
    return tstdata['class'],predictions 


# In[8]:

print "reading datasets.....\n"
x,y=read_datasets()
x.describe()


# In[9]:

print "extracting featues.....\n"
x=extract_features(x)
print x.columns
print x.describe()


# In[10]:

print "training datasets.......\n"
y_test,y_pred =train(x,y)


# In[11]:

print 'Classification Accuracy on Test dataset: ' ,accuracy_score(y_test, y_pred)


# In[12]:

print 'Percent Error on Test dataset: ' ,percentError(y_pred,y_test)


# In[13]:

cm=confusion_matrix(y_test, y_pred)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)


# In[14]:

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# In[15]:

print(classification_report(y_test, y_pred, target_names=['Fake','Genuine']))


# In[16]:

s=roc_auc_score(y_test, y_pred)
print "roc_auc_score : ",s


# In[17]:

plot_roc_curve(y_test, y_pred)

