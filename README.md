# Particle physics event classification

This is a supervised machine learning model for a [particle physics event classification dataset](https://www.kaggle.com/datasets/younusmohamed/particle-physics-event-classification-dataset?select=README.md).


## Preprocessing the data

The dataset contains 250,000 rows (events) and 33 columns (an event ID, 30 feature variables, a target variable, and a weight variable). Each event is classified as either a signal (s) or background (b), which is the basis of prediction for our supervised learning model.

We first perform an 80-20 split of the data into a training set and a testing set. The feature medians from the training set are used to impute the missing values in both sets. Similarly, the feature means and standard deviations from the training set are used to scale the values in both sets via z-score normalisation. This ensures that there is no data leakage.

Note: The dataset creator recommends passing an extra column labeled 'Weight' to the `sample_weight` parameter during training and testing as an indicator of the experimental importance of events. But one cannot know _a priori_ which experiments will be important. For this reason, we do not use this column. An alternative approach is to view this column as the target variable for a regression-based model.


## Analysing and selecting features




## Developing the model
