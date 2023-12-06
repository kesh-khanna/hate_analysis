# Load the UC Berkeley Hate Speech dataset and preprocess it

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import datasets
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# import linear svm
from sklearn.svm import LinearSVC


def download_data():
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
    print(dataset)
    df = dataset['train'].to_pandas()
    df.describe()
    return df

def preprocess():
    df = download_data()
    # keep the columns we need
    df = df[['text', 'hate_speech_score', 'annotator_severity']]

    # remove rows with NaN
    df = df.dropna()

    print(df.shape)

    # subtract the annotator severity from the hate speech score
    # df['hate_speech_score'] = df['hate_speech_score'] + df['annotator_severity']

    # remove rows with a negative hate speech score
    # df = df[df['hate_speech_score'] >= -1]

    print(df.shape)

    # split the data into offensive and hate speech, add new binary column
    # leave small buffer between offensive and hate speech to help with classification
    df_offensive = df[df['hate_speech_score'] < 0.5]
    df_offensive.loc[:, 'label'] = 0
    df_hate = df[df['hate_speech_score'] > 0.5]
    df_hate.loc[:, 'label'] = 1

    # combine the two dataframes
    df = pd.concat([df_offensive, df_hate])
    print(df_offensive.shape)
    print(df_hate.shape)
    print(df.shape)

    print(df['label'].value_counts())

    # graph the distribution of the hate speech score
    plt.hist(df['hate_speech_score'], bins=20)
    plt.title('Distribution of Hate Speech Score over 0')
    plt.savefig('hate_speech_score.png')

    return df


def main():
    
    df = preprocess()
    # create our pipeline
    # remove punctuation, lowercase, tokenize, remove stopwords, lemmatize
    # vectorize using TF-IDF
    # split into train and test sets

    model = Pipeline([("vector", CountVectorizer(ngram_range=(1, 3), stop_words='english')),
                      ("trans", TfidfTransformer()),
                      ("clf", LinearSVC(penalty="l2", C=1))])

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # check the distribution of the predictions
    plt.figure()
    plt.hist(y_pred, bins=20)
    plt.title('Distribution of Predictions')
    plt.savefig('predictions.png')

    misclassified = y_test[y_pred != y_test]

    print(misclassified.shape)

    # plot the distribution of the incorrectly classified examples
    plt.figure()
    plt.hist(misclassified, bins=20)
    plt.title('Distribution of Incorrectly Classified Examples')
    plt.savefig('incorrectly_classified.png')

    # look at correctly classified examples
    correctly_classified = X_test[y_pred == y_test]
    correctly_classified_y = y_test[y_pred == y_test]

    print(correctly_classified.shape)
    print(correctly_classified_y.shape)

    # plot the distribution of the correctly classified examples
    plt.figure()
    plt.hist(correctly_classified_y, bins=20)
    plt.title('Distribution of Correctly Classified Examples')
    plt.savefig('correctly_classified.png')







# Custom transformer to select numerical columns
class NumericalSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[['hate_speech_score']]


# Custom transformer to select text column
class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X['text']


columns = ['text', 'hate_speech_score', 'annotator_ideology', 'respect', "insult", 'sentiment', 'humiliate', "status",
           "dehumanize", "violence", "genocide", "attack_defend"]


def predict_ideology():
    """
    Predict the ideology of the annotators based on given hate speech scores and text
    """
    df = download_data()

    print(df.shape)

    # remove rows with NaN
    df = df.dropna()

    # preprocessor to split the text and numerical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TextSelector(), 'text'),
            ('num', NumericalSelector(), ['hate_speech_score']),
        ])

    # create our pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('features', FeatureUnion([
            ('text', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
            ('num_scaler', StandardScaler())
        ])),
        ('classifier', RandomForestClassifier())
    ])

    X = df[["text", "hate_speech_score"]]
    y = df['annotator_ideology']

    # encode the y
    le = LabelEncoder()
    y = le.fit_transform(y)

    # one hot encode the y
    y = pd.get_dummies(y)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(y_train.shape)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
    # predict_ideology()
