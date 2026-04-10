import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import analysis
from model import Model


if __name__ == '__main__':
    sns.set_style('darkgrid')


    # DATA PREPROCESSING

    df = pd.read_csv('data.csv')
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

    y_train_binary = y_train.map(lambda x: 1 if x == 's' else 0)

    feat_median_diffs = analysis.median_differences(
        X_train_scal, y_train_binary, plot_top_n=10, left_adjust=0.32
    )
    feat_target_corr = analysis.feature_target_correlation(
        X_train_scal, y_train_binary, plot_top_n=10, left_adjust=0.32
    )
    feat_feat_corr = analysis.feature_feature_correlation(
        X_train_scal, plot_top_n=15, adjust=0.25
    )
    feat_corr = analysis.drop_highly_intercorrelated(
        feat_target_corr, feat_feat_corr, threshold=0.75
    )

    feat_by_median = feat_median_diffs.iloc[:10].index
    feat_by_corr = feat_corr[feat_corr > 0.15].index

    feat = feat_by_median.union(feat_by_corr).to_list()
    print('\nFeatures selected for the ML model')
    print(pd.Series(feat))

    X_train_sel = X_train_scal[feat]
    X_test_sel = X_test_scal[feat]


    # MODEL TRAINING AND TESTING

    model = Model(X_train_sel, X_test_sel, y_train, y_test, pos_label='s')

    estimator = KNeighborsClassifier()

    best_params = model.tune_parameters(
        estimator=estimator,
        params={
            'n_neighbors': range(35, 65, 2),
            'p': [1]
        },
        scoring='roc_auc',
        seed=13
    )
    estimator = estimator.set_params(**best_params)

    model.train_and_test(estimator)
