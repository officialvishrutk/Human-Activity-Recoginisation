import csv
from sklearn import metrics
from numba import jit
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_random_state

#@jit(nopython=True)
def fun():
    with open('labels.csv','r') as dest_f:
        lb = list(csv.reader(dest_f, delimiter = ',' ))
    print('labels.csv File Opened Sucessfully')

    with open('images.csv','r') as dest_f: 
        img = list(csv.reader(dest_f, delimiter = ',' ))
    print('images.csv File Opened Sucessfully')

    X_train, X_test, y_train, y_test = train_test_split(img, lb, test_size=0.2)

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=46)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    # print('Confusion Matrix:')
    # print(metrics.confusion_matrix(y_test, y_pred))

    #confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    
    featureImportances = pd.Series(clf.feature_importances_).sort_values(ascending=False)
    print(featureImportances)
    sn.barplot(x = round(featureImportances, 4), y = featureImportances)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    plt.show()

fun()