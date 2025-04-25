import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from datetime import datetime,timezone

"""Reading Datasets from csv files"""

def read_datasets():
  genuine_users= pd.read_csv("users (1).csv")
  fake_users= pd.read_csv("fusers (1).csv")
  print(genuine_users.columns)
  print(genuine_users.describe())
  print(fake_users.describe())
  x=pd.concat([genuine_users, fake_users])
  y=len(fake_users)*[0] + len(genuine_users)*[1]
  return x,y

print("reading datasets.....\n")
x,y=read_datasets()
x.describe()

def extract_features(x):
  # Ensure 'created_at' is in datetime format
  x['created_at'] = pd.to_datetime(x['created_at'], errors='coerce')

  # Calculate account age in days
  current_time = datetime.now(timezone.utc)
  x['account_age'] = (current_time - x['created_at']).dt.days

  # Select features
  feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'account_age']
  x = x.loc[:, feature_columns_to_use]

  # Handle missing values, if any
  x = x.fillna(0)

  return x
print("extracting featues.....\n")
x=extract_features(x)
print(x)

print("spliting datasets in train and test dataset...\n")
X_train, X_test,y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=44)
print(X_train)
print(X_test)

def train (X_train, y_train, X_test):
  clf=RandomForestClassifier(n_estimators=40,oob_score=True)
  clf.fit(X_train,y_train)
  print("The classifier is: ",clf)
#Predict
  y_pred = clf.predict(X_test)
  return y_test,y_pred,clf

print("training datasets\n")
y_test, y_pred, clf = train(X_train,y_train,X_test)
print(clf)

def plot_confusion_matrix(cm,title='Confusion matrix',cmap=plt.cm.Blues):
  target_names=['Fake', 'Genuine']
  plt.imshow(cm,interpolation='none', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(target_names))
  plt.xticks(tick_marks, target_names, rotation=45)
  plt.yticks (tick_marks, target_names)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

cm=confusion_matrix(y_test, y_pred)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)
cm_normalized=cm.astype('float')/ cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

#finding Accuracy score
X_test_prediction=clf.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)
print('Accuracy Score of Test data:',test_data_accuracy)

def plot_roc_curve(y_test, y_pred):
  false_positive_rate, true_positive_rate, thresholds=roc_curve(y_test, y_pred)
  print("False Positive rate: ",false_positive_rate)
  print("True Positive rate: ", true_positive_rate)
  roc_auc=auc(false_positive_rate,true_positive_rate)
  plt.title('Receiver Operating Characteristic Curve')
  plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1], [0,1], 'r--')
  plt.xlim([-0.1,1.2])
  plt.ylim([-0.1,1.2])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()
print('Classification Accuracy on Test dataset:',accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, target_names=['Fake' ,'Genuine']))
plot_roc_curve(y_test, y_pred)





