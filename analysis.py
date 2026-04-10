import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(X_train, X_test, imputer, scaler):
    imputer.set_output(transform='pandas')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler.set_output(transform='pandas')
    X_train_scal = scaler.fit_transform(X_train_imp)
    X_test_scal = scaler.transform(X_test_imp)

    return X_train_imp, X_test_imp, X_train_scal, X_test_scal


def _series_barplot(series, title, diverging=False, left_adjust=0):
    palette = 'RdBu' if diverging else 'crest'

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x=series.values, y=series.index,
        hue=series.values, palette=palette, legend=False
    )

    ax.set(title=title, xlabel=None, ylabel=None)
    if left_adjust:
        plt.subplots_adjust(left=left_adjust)

    plt.savefig(f'vis/{title.replace(' ', '_')}.png')
    plt.show()


def mean_differences(X_train, y_train, plot_top_n=0, left_adjust=0):
    mean_0 = X_train[y_train == 0].mean()
    mean_1 = X_train[y_train == 1].mean()
    feat_mean_diffs = (mean_1 - mean_0).abs().sort_values(ascending=False)

    if plot_top_n:
        _series_barplot(
            series=feat_mean_diffs[:plot_top_n],
            title=f'Top {plot_top_n} mean differences between classes',
            left_adjust=left_adjust
        )

    return feat_mean_diffs


def median_differences(X_train, y_train, plot_top_n=10, left_adjust=0):
    median_0 = X_train[y_train == 0].median()
    median_1 = X_train[y_train == 1].median()
    feat_median_diffs = (median_1 - median_0).abs().sort_values(ascending=False)

    if plot_top_n:
        _series_barplot(
            series=feat_median_diffs[:plot_top_n],
            title=f'Top {plot_top_n} median differences between classes',
            left_adjust=left_adjust
        )

    return feat_median_diffs


def feature_target_correlation(X_train, y_train, plot_top_n=10, left_adjust=0):
    feat_target_corr = X_train.agg(y_train.corr).sort_values(
        ascending=False,
        key=lambda x: x.abs()
    )

    if plot_top_n:
        _series_barplot(
            series=feat_target_corr[:plot_top_n],
            title=f'Top {plot_top_n} feature-target correlations',
            diverging=True,
            left_adjust=left_adjust
        )

    return feat_target_corr


def feature_feature_correlation(X_train, plot_top_n=15, adjust=0):
    corr_matrix = X_train.corr()

    feat_feat_corr = corr_matrix.stack().reset_index()
    feat_feat_corr.columns = ['feature_1', 'feature_2', 'correlation']
    feat_feat_corr = feat_feat_corr[
        feat_feat_corr['feature_1'] != feat_feat_corr['feature_2']
    ].sort_values(
        'correlation',
        ascending=False,
        key=lambda x: x.abs()
    )
    top_n_corr = feat_feat_corr[:2*plot_top_n] # due to matrix symmetry
    high_corr_feat = top_n_corr['feature_1'].drop_duplicates(ignore_index=True)
    high_corr_matrix = corr_matrix.loc[high_corr_feat, high_corr_feat]

    if plot_top_n:
        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(
            high_corr_matrix,
            vmin=-1, vmax=1, center=0,
            cmap='RdBu'
        )

        ax.set(
            title=f'Top {plot_top_n} feature-feature correlations',
            xlabel=None, ylabel=None
        )
        if adjust:
            plt.subplots_adjust(left=adjust, bottom=adjust)

        plt.savefig(f'vis/Top_{plot_top_n}_feature-feature_correlations.png')
        plt.show()

    return feat_feat_corr


def drop_highly_intercorrelated(feat_prop, feat_feat_corr, threshold=0.9):
    high_corr_feat = feat_feat_corr[feat_feat_corr['correlation'] >= threshold]

    drop_feat = high_corr_feat.apply(
        lambda row: row['feature_1']
        if feat_prop[row['feature_1']] < feat_prop[row['feature_2']]
        else row['feature_2'],
        axis=1
    ).drop_duplicates()

    return feat_prop.drop(index=drop_feat)
