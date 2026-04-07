import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
    StratifiedKFold)
from sklearn import metrics


class Model:
    def __init__(self, X_train, X_test, y_train, y_test, pos_label):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.pos_label = pos_label
        self.neg_label = y_train[y_train != pos_label].iat[0]


    def _score(self, y_prob, threshold=0.5, plot_roc_curve=False):
        y_pred = pd.Series(y_prob[:, 1]).map(
            lambda p: self.pos_label if p >= threshold else self.neg_label
        )

        print('\nConfusion matrix')
        print(metrics.confusion_matrix(self.y_test, y_pred))

        print('\nClassification report')
        print(metrics.classification_report(self.y_test, y_pred))

        print('ROC AUC score:',
            metrics.roc_auc_score(self.y_test, y_prob[:, 1]))

        if plot_roc_curve:
            fpr, tpr, threshold = metrics.roc_curve(
                self.y_test, y_prob[:, 1], pos_label=self.pos_label
            )
            ax = sns.lineplot(x=range(2), y=range(2))
            sns.lineplot(x=fpr, y=tpr)
            ax.set(
                title='ROC Curve',
                xlabel='False Positive Rate',
                ylabel='True Positive Rate'
            )
            plt.show()


    def train_and_test(self, estimators, threshold=0.5, plot_roc_curve=False):
        try:
            for estimator in estimators:
                print('\nModel:', estimator)
                estimator.fit(self.X_train, self.y_train)
                y_prob = estimator.predict_proba(self.X_test)
                self._score(y_prob, threshold, plot_roc_curve)

        except TypeError:
            print('\nModel:', estimators)
            estimators.fit(self.X_train, self.y_train)
            y_prob = estimators.predict_proba(self.X_test)
            self._score(y_prob, threshold, plot_roc_curve)


    def tune_parameters(self, estimator, params, scoring, grid=False, seed=0):
        if seed:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        else:
            kf = StratifiedKFold(n_splits=5, shuffle=True)

        if grid:
            cv = GridSearchCV(estimator, params, scoring=scoring, cv=kf)
        else:
            cv = RandomizedSearchCV(estimator, params, scoring=scoring, cv=kf)

        cv.fit(self.X_train, self.y_train)

        print(f'\n{'Best parameters':<20} {cv.best_params_}')
        print(f'\n{'Best score':<20} {cv.best_score_}')
