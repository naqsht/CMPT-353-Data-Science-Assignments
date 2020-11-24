import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


OUTPUT_TEMPLATE = ('Score: {score:.3g}\n')

def main():

    # Reading command line arguments
    monthly_data_labelled = pd.read_csv(sys.argv[1])
    monthly_data_unlabelled = pd.read_csv(sys.argv[2])

    X = monthly_data_labelled.loc[:,'tmax-01':'snwd-12'].values
    y = monthly_data_labelled['city'].values

    # Split train and test data
    X_train, X_valid, y_train, y_valid = train_test_split(X,y)

    # Construct SVM Classifier
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.1))

    # Fitting the model
    svm_model.fit(X_train, y_train)

    # Predicting Cities
    X_unlabelled = monthly_data_unlabelled.loc[:,'tmax-01':'snwd-12'].values
    predictions = svm_model.predict(X_unlabelled)

    pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)


    # Print model score
    print(OUTPUT_TEMPLATE.format(score=svm_model.score(X_valid, y_valid)))


main()