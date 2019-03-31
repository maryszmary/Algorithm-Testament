import pickle

class FunnySentences:
    def __init__(self, markov_chain_path, classifier_path):
        with open(markov_chain_path, 'rb') as f:
            self.lang_model = pickle.load(f)
        with open(classifier_path, 'rb') as f:
            self.clf_model = pickle.load(f)
        self.min_p = 0.15
        self.max_p = 0.85

    def is_funny(self, sent):
        return self.min_p < self.clf_model.predict_proba([sent])[0][1] < self.max_p

    def gen_funny(self, n, substr=''):
        funny = []
        for i in range(300 * int(n/10)):
            sent = self.lang_model.make_sentence()
            if self.is_funny(sent) and len(sent) < 140 and substr in sent:
                funny.append(sent)
            if len(funny) == n:
                return funny
        return funny        