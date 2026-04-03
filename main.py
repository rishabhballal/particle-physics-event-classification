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
from sklearn import metrics


def select_feat_by_median(X_train, y_train, n=10):
    median_s = X_train[y_train == 's'].median()
    median_b = X_train[y_train == 'b'].median()
    features = (median_s - median_b).abs().nlargest(n)
    if not features.empty:
        print(f'Classification differences in median (top {n})\n')
        print(features.reset_index().to_string(header=False))
    return features


def select_feat_by_correlation(X_train, y_train, n=10, threshold=0.75):
    # feature-target linear correlations
    y_train_binary = y_train.map(lambda x: 1 if x == 's' else 0)
    feat_target_corr = X_train.agg(y_train_binary.corr).abs().nlargest(n)
    print(f'\nFeature-target correlation (top {n})')
    print(feat_target_corr.reset_index().to_string(header=False))

    # feature-feature linear correlations
    feat_corr_matrix = X_train[feat_target_corr.index].corr()
    feat_corr_matrix = pd.DataFrame(
        np.triu(feat_corr_matrix.values),
        index=feat_corr_matrix.index,
        columns=feat_corr_matrix.columns
    )
    high_corr_feat = feat_corr_matrix.stack().reset_index()
    high_corr_feat.columns = ['feature_1', 'feature_2', 'correlation']
    high_corr_feat = high_corr_feat[
        (high_corr_feat['correlation'].abs() > threshold)
        & (high_corr_feat['feature_1'] != high_corr_feat['feature_2'])
    ].reset_index(drop=True)
    if not high_corr_feat.empty:
        print(f'\nFeature-feature correlation (threshold = {threshold})')
        print(high_corr_feat.to_string(header=False))

    # unselect strongly correlated features
    drop_feat = high_corr_feat.apply(
        lambda row: row['feature_1'] if feat_target_corr[row['feature_1']]
        < feat_target_corr[row['feature_2']] else row['feature_2'],
        axis=1
    ).drop_duplicates()
    if not drop_feat.empty:
        features = feat_target_corr.drop(index=drop_feat)
        print('\nFeatures dropped from feature-target correlation table')
        print(drop_feat.reset_index(drop=True))
    else:
        features = feat_target_corr

    return features


def classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('\nConfusion matrix')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('\nClassification report')
    print(metrics.classification_report(y_test, y_pred))

    y_proba = clf.predict_proba(X_test)
    print('ROC AUC score:', metrics.roc_auc_score(y_test, y_proba[:, 1]))
    fpr, tpr, threshold = metrics.roc_curve(
        y_test, y_proba[:, 1], pos_label='s'
    )
    ax = sns.lineplot(x=range(2), y=range(2))
    sns.lineplot(x=fpr, y=tpr)
    ax.set(
        title='ROC Curve',
        xlabel='False Positive Rate',
        ylabel='True Positive Rate'
    )
    plt.show()


if __name__ == '__main__':
    sns.set_style('darkgrid')
    df = pd.read_csv('particle_physics_data.csv')


    # DATA CLEANING

    df = df.replace(-999, np.nan)

    X = df.drop(columns=['EventId', 'Weight', 'Label'])
    y = df['Label'] # target for classification
    w = df['Weight'] # target for regression

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.20, shuffle=True, stratify=y, random_state=13
    )

    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # restore DataFrame structure for simpler manipulation
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = y_train.reset_index(drop=True)


    # FEATURE SELECTION

    feat_by_median = select_feat_by_median(X_train, y_train)
    feat_by_corr = select_feat_by_correlation(X_train, y_train)

    feat = list(set(pd.concat([feat_by_median, feat_by_corr]).index))
    print('\nFeatures selected by median differences and target correlation')
    print(pd.Series(feat))

    X_train = X_train[feat]
    X_test = X_test[feat]


    # MODEL TRAINING

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=13, p=1)
    dtree = DecisionTreeClassifier(criterion='entropy')
    logreg = LogisticRegression()
    classifier(knn, X_train_scaled, X_test_scaled, y_train, y_test)
