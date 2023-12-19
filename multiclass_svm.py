"""
Multiclass implementation of same SVM procedure as in preprocessing.py
"""

from preprocessing import download_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


def preprocess():
    df = download_data()
    # keep the columns we need
    df = df[['text', 'hate_speech_score', 'annotator_severity']]
    # drop rows with missing data
    df = df.dropna()
    # create a new column for the label below -1 is supportive -1 to 0.5 is neutral above 0.5 is hateful
    df['label'] = df['hate_speech_score'].apply(lambda x: 0 if x < -1 else 1 if x < 0.5 else 2)

    print(df['label'].value_counts())

    return df


def svm():
    # load data

    df = preprocess()

    # split data
    X = list(df['text'])
    y = list(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create pipeline
    model = Pipeline([("vector", CountVectorizer(ngram_range=(1, 3), stop_words='english')),
                      ("transformer", TfidfTransformer()),
                      ("classifier", LinearSVC(C=1.0, penalty="l2", dual="auto", multi_class="crammer_singer"))])

    # train model
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def inherent_multiclass():
    """
    Test out Random Forest for an inherent multiclass implementation (not one-vs-rest)
    """
    df = preprocess()

    X = list(df['text'])
    y = list(df['label'])

    # one hot encode the labels
    y = pd.get_dummies(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Building Pipeline")
    model = Pipeline([("vector", CountVectorizer(ngram_range=(1, 3), stop_words='english')),
                      ("transformer", TfidfTransformer()),
                      ("classifier", RandomForestClassifier(n_estimators=5, verbose=1))])

    print("Training Model")
    model.fit(X_train, y_train)

    print("Evaluating Model")
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # plot confusion matrix
    cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)


def dist_fig():
    """
    make a fig to show the distribution of the labels
    """
    df = preprocess()

    X = list(df['text'])
    y = list(df['label'])

    # make seaborn distplot
    sns.histplot(y, bins=3)
    plt.title('Distribution of Labels')

    # add x labels
    plt.xticks(np.arange(3), ['Supportive', 'Neutral', 'Hateful'])

    plt.savefig('label_dist.png')


if __name__ == '__main__':
    svm()
    # inherent_multiclass()

    # dist_fig()
