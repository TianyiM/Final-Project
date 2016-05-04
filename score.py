

import textprocessing as tp
import pickle
import itertools
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score



pos_review = tp.seg_fil_senti_excel("~", 1, 1)
neg_review = tp.seg_fil_senti_excel("~", 1, 1)

pos = pos_review
neg = neg_review


def bag_of_words(words):
    return dict([(word, True) for word in words])


def bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(bigrams)


def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)



def create_word_scores():
    posdata = tp.seg_fil_senti_excel("~", 1, 1)
    negdata = tp.seg_fil_senti_excel("~", 1, 1)
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    word_fd = FreqDist()
    last_word = ConditionalFreqDist()
    for word in posWords:
        word_fd.inc(word)
        last_word['pos'].inc(word)
    for word in negWords:
        word_fd.inc(word)
        last_word['neg'].inc(word)

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(last_word['pos'][word], (freq, pos_word_count), totalnumber)
        neg_score = BigramAssocMeasures.chi_sq(last_word['neg'][word], (freq, neg_word_count), totalnumber)
        word_scores[word] = pos_score + neg_score

    return word_scores

def ctreat_scores():
    posdata = tp.seg_fil_senti_excel("~", 1, 1)
    negdata = tp.seg_fil_senti_excel("~", 1, 1)
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    pos = posBigrams
    neg = negBigrams

    word_fd = FreqDist()
    last_word = ConditionalFreqDist()
    for word in pos:
        word_fd.inc(word)
        last_word['pos'].inc(word)
    for word in neg:
        word_fd.inc(word)
        last_word['neg'].inc(word)


    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(last_word['pos'][word], (freq, pos_word_count), totalnumber)
        neg_score = BigramAssocMeasures.chi_sq(last_word['neg'][word], (freq, neg_word_count), totalnumber)
        word_scores[word] = pos_score + neg_score

    return word_scores

def create_word_bigram_scores():
    posdata = tp.seg_fil_senti_excel("~", 1, 1)
    negdata = tp.seg_fil_senti_excel("~", 1, 1)
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams
    neg = negWords + negBigrams

    word_fd = FreqDist()
    last_word = ConditionalFreqDist()
    for word in pos:
        word_fd.inc(word)
        last_word['pos'].inc(word)
    for word in neg:
        word_fd.inc(word)
        last_word['neg'].inc(word)

    pos_word_count = last_word['pos'].N()
    neg_word_count = last_word['neg'].N()
    totalnumber = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(last_word['pos'][word], (freq, pos_word_count), totalnumber)
        neg_score = BigramAssocMeasures.chi_sq(last_word['neg'][word], (freq, neg_word_count), totalnumber)
        word_scores[word] = pos_score + neg_score

    return word_scores


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


def best_word_features_bi(words):
    return dict([(word, True) for word in nltk.bigrams(words) if word in best_words])

def best_word_features_com(words):
    d_1 = dict([(word, True) for word in words if word in best_words])
    d_2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d_3 = dict(d_1, **d_2)
    return d_3

def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i),'pos']
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j),'neg']
        negFeatures.append(negWords)
    return negFeatures


best_words = find_best_words(word_scores, 1500) 



posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)

shuffle(posFeatures)
shuffle(negFeatures)

size_pos = int(len(pos_review) * 0.75)
size_neg = int(len(neg_review) * 0.75)

train_set = posFeatures[:size_pos] + negFeatures[:size_neg]
test_set = posFeatures[size_pos:] + negFeatures[size_neg:]

test, tag_test = zip(*test_set)

def clf_score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)

    predict = classifier.batch_classify(test)
    return accuracy_score(tag_test, predict)


def score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(trainset)

    pred =
for d in feature_number:
    word_scores = create_word_bigram_scores()
    best_words = find_best_words(word_scores, int(d))

    posFeatures = pos_features(best_word_features_com)
    negFeatures = neg_features(best_word_features_com)
 classifier.batch_classify(test)
    return accuracy_score(tag_test, pred)

feature_number = []

  
def store_classifier(clf, trainset, filepath):
    classifier = SklearnClassifier(clf)
    classifier.train(trainset)
    pickle.dump(classifier, open(filepath,'w'))

dev, tag_dev = zip(*devtest)

def score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(train) 

    pred = classifier.batch_classify(testSet)
    return accuracy_score(tag_test, pred)

import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

posFeatures = pos_features(bigrams)
negFeatures = neg_features(bigrams)

word_scores = create_word_scores()
best_words = find_best_words(word_scores, 1500)


posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)

import pickle
import sklearn


clf = pickle.load(open('~'))

pred = clf.batch_prob_classify(moto_features)
p_file = open('~','w')
for i in pred:
    p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
p_file.close()
