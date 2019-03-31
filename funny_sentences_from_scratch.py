import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from string import punctuation

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


class FunnySentences:
    def __init__(self, t1, t2):
        self.testament = sent_tokenize(t1)
        self.coding = sent_tokenize(t2)
        self.sentences = self.testament + self.coding
        labels = ['testament'] * len(self.testament) + ['coding'] * len(self.coding)
        df = pd.DataFrame({'sentences' : self.sentences, 'labels': labels})
        df = df.sample(frac=1).reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(
                df['sentences'], df['labels'], test_size=0.2, random_state=42
                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.stopset = stopwords.words('russian')
        self.punct = list(punctuation)
        self.min_p = 0.15
        self.max_p = 0.85
        pass

    def preprocessor(self):
        def ret(text):
            text = word_tokenize(text.lower())
            return [w for w in text if w not in self.stopset + self.punct]
        return ret

    def learn(self):
        self.model = Pipeline(
            [('vect', TfidfVectorizer(min_df=5, tokenizer=self.preprocessor())),
            ('nb', MultinomialNB())]
        )
        self.model.fit(self.X_train, y=self.y_train)
        print(classification_report(self.y_test, self.model.predict(self.X_test)))
        print(accuracy_score(self.y_test, self.model.predict(self.X_test)))

    def is_funny(self, sent):
        try:
            return self.min_p < self.model.predict_proba([sent])[0][1] < self.max_p
        except:
            raise Exception("You should run «learn» method first")

    def gen_funny(self, generator, n, substr=''):
        funny = []
        for i in range(300 * int(n/10)):
            sent = generator()
            if self.is_funny(sent) and len(sent) < 140 and substr in sent:
                funny.append(sent)
            if len(funny) == n:
                return funny
        return funny