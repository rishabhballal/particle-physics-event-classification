import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import analysis
from model import Model


if __name__ == '__main__':
    sns.set_style('darkgrid')


    # DATA PREPROCESSING

    df = pd.read_csv('particle_physics_data.csv')
    df = df.replace(-999, np.nan)

    X = df.drop(columns=['EventId', 'Weight', 'Label'])
    y = df['Label'] # target for classification
    w = df['Weight'] # target for regression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, stratify=y, random_state=13
    )

    X_train_imp, X_test_imp, X_train_scal, X_test_scal = analysis.preprocess(
        X_train, X_test,
        SimpleImputer(strategy='median'),
        StandardScaler()
    )


    # FEATURE SELECTION

    feat_by_median = analysis.select_feat_by_median(X_train_imp, y_train)
    feat_by_corr = analysis.select_feat_by_correlation(X_train_imp, y_train)

    feat = list(set(pd.concat([feat_by_median, feat_by_corr]).index))
    print('\nFeatures selected by median differences and target correlation')
    print(pd.Series(feat))

    X_train_scal_ = X_train_scal[feat]
    X_test_scal_ = X_test_scal[feat]


    # MODEL TRAINING AND TESTING

    model = Model(X_train_scal_, X_test_scal_, y_train, y_test)
    model.train_and_test(KNeighborsClassifier(n_neighbors=13, p=1))
