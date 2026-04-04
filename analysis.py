import numpy as np
import pandas as pd


def preprocess(X_train, X_test, imputer, scaler):
    imputer.set_output(transform='pandas')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler.set_output(transform='pandas')
    X_train_scal = scaler.fit_transform(X_train_imp)
    X_test_scal = scaler.transform(X_test_imp)

    return X_train_imp, X_test_imp, X_train_scal, X_test_scal


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
