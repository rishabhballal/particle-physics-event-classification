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

        return cv.best_estimator_


    def confusion_matrix(self, y_pred):
        matrix = pd.DataFrame(
            metrics.confusion_matrix(self.y_test, y_pred),
            index=['Actual negative', 'Actual positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            matrix,
            vmin=0,
            vmax=len(self.y_test),
            cmap='Blues',
            annot=True,
            fmt='d'
        )
        ax.set_title('Confusion matrix')

        plt.savefig('vis/Confusion_matrix.png')
        plt.show()


    def classification_report(self, y_pred):
        report = metrics.classification_report(
            self.y_test, y_pred, output_dict=True
        )

        report_df = pd.DataFrame(
            {self.neg_label: report[self.neg_label].values(),
            self.pos_label: report[self.pos_label].values()},
            index=report[self.pos_label].keys()
        ).drop(index='support').stack().reset_index()

        report_df.columns = ['Metric', 'Label', 'Score']

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            report_df,
            x='Metric',
            y='Score',
            hue='Label'
        )

        xlabels = [x.get_text()[0].upper() + x.get_text()[1:]
            for x in ax.get_xticklabels()]
        ax.set(
            title='Classification report',
            xlabel=None,
            ylabel=None,
            xticks=[0, 1, 2],
            xticklabels=xlabels,
            yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )

        plt.savefig('vis/Classification_report.png')
        plt.show()


    def roc_curve(self, y_prob):
        auc = metrics.roc_auc_score(self.y_test, y_prob[:, 1])

        fpr, tpr, threshold = metrics.roc_curve(
            self.y_test, y_prob[:, 1], pos_label=self.pos_label
        )

        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(x=fpr, y=tpr)
        sns.lineplot(
            x=[0, 1], y=[0, 1],
            color='C7', linestyle='dashed', alpha=0.4
        )

        ax.set(
            title='ROC curve',
            xlabel='False positive rate',
            ylabel='True positive rate'
        )
        ax.text(
            x=0.7, y=0.3,
            s=f'AUC = {auc:.2f}',
            ha='center', va='center'
        )

        curve = list(ax.get_lines())[0].get_xydata().T
        ax.fill_between(
            x=curve[0],
            y1=0,
            y2=curve[1],
            color='C0',
            alpha=0.4
        )

        plt.savefig('vis/ROC_curve.png')
        plt.show()


    def train_and_test(self, estimator, threshold=0.5):
        print('\nModel:', estimator)
        estimator.fit(self.X_train, self.y_train)

        y_prob = estimator.predict_proba(self.X_test)
        y_pred = pd.Series(y_prob[:, 1]).map(
            lambda p: self.pos_label if p >= threshold else self.neg_label
        )

        self.confusion_matrix(y_pred)
        self.classification_report(y_pred)
        self.roc_curve(y_prob)
