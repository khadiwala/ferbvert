from itertools import chain
from sklearn.svm import LinearSVC
from sklearn import feature_extraction as fe
import numpy as np

load = lambda x: open(x).readlines()
stopwords = set(load("english"))
stopwords.add("twss")
stopwords.add("fml")


def tokens(s):
    return [word for word in s.rstrip().lower().split() if word not in stopwords]


"""
from time import time
def benchmark(clf, X, Y, train, test):
    clf_descr = clf[0]
    clf = clf[1]
    print(80 * '_')
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X[train], Y[train])
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X[test])
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(Y[test], pred)
    print("f1-score:   %0.3f" % score)

    print("confusion matrix:")
    print(metrics.confusion_matrix(Y[test], pred))

    print()
    return clf_descr, score, train_time, test_time


cv = cross_validation.KFold(all_h.shape[0], n_folds=5, shuffle=True)
 train,test = next(iter(cv)) # just use one of the folds
clss = [         ("LinearSVC l1", LinearSVC(loss='l2', penalty="l1", dual=False, tol=1e-3))        ]
results = [benchmark(cls,all_h,target,train,test) for cls in clss]
"""


def get_classifier():
    pos = load("data/twss-stories-parsed.txt")
    negs = ["data/fmylife-parsed.txt", "data/texts-from-last-night-parsed.txt"]
    neg = list(chain(*map(load, negs)))
    pos_f = map(tokens, pos)
    neg_f = map(tokens, neg)
    all_f = chain(pos_f, neg_f)
    target = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    fh = fe.FeatureHasher(input_type='string')
    all_h = fh.transform(all_f)
    clf = LinearSVC(loss='squared_hinge', penalty="l1", dual=False, tol=1e-3)
    clf.fit(all_h, target)

    def classify(s):
        return clf.predict(fh.transform([tokens(s)]))[0] == 1

    return classify


if __name__ == '__main__':
    cl = get_classifier()
    print cl("harder than deeper than more words soft")
    print cl("negative example")
