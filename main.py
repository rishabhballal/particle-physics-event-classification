import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def classifier(clf, X_train, X_test, y_train, y_test, w_train, w_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))
    print('\n', metrics.classification_report(y_test, y_pred))

    y_proba = clf.predict_proba(X_test)
    print('ROC AUC score:', metrics.roc_auc_score(y_test, y_proba[:, 1]))
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_proba[:, 1], pos_label='s')
    fig, ax = plt.figure()
    ax.plot(range(2), range(2))
    ax.plot(fpr, tpr)
    ax.title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('particle_physics_data.csv')

    # identifying NaNs
    df = df.replace(-999, np.nan)

    # categorising the columns into features and target(s)
    X = df.drop(columns=['EventId', 'Weight', 'Label'])
    y = df['Label'] # for classification
    w = df['Weight'] # for regression

    # splitting the data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.20, shuffle=True, stratify=y
    )

    # imputing the data
    imputer = SimpleImputer()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # normalising the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # restore DataFrame structure
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = y_train.reset_index(drop=True)

    # feature summary statistics
    agg = X_train.agg([np.min, np.mean, np.median, np.max, np.var]).T

    # training the model
    knn = KNeighborsClassifier(n_neighbors=1, p=2)
    dtree = DecisionTreeClassifier(criterion='gini')
    logreg = LogisticRegression()
    classifier(knn, X_train, X_test, y_train, y_test)
