#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import string
import numpy as np
import time
import math


# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos):
    llist = wn.lemmas(lemma, pos=pos)
    list1 = [x.synset().lemmas() for x in llist]
    list2 = [x.name() for y in list1 for x in y]
    list3 = [str.replace('_', ' ') for str in list2]
    return set(list3) - {lemma.replace('_', ' ')}


def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context):
    possible_synonyms = {}
    llist = wn.lemmas(context.lemma, pos=context.pos)
    list1 = [x.synset().lemmas() for x in llist]
    list2 = [(x.name().replace('_', ' '), x.count()) for y in list1 for x in y]
    for i in list2:
        if i[0] == context.lemma.replace('_', ' '):
            continue
        if i[0] in possible_synonyms:
            possible_synonyms[i[0]] += i[1]
        else:
            possible_synonyms[i[0]] = i[1]
    return max(possible_synonyms, key=possible_synonyms.get)


def wn_simple_lesk_predictor(context):
    llist = wn.lemmas(context.lemma, pos=context.pos)
    list1 = [x.synset() for x in llist]
    stop_words = stopwords.words('english')
    conl = [tokenize(x) for x in context.left_context]
    conr = [tokenize(x) for x in context.right_context]
    tlist = [x for j in (conl + conr + [tokenize(context.lemma)]) for x in j]
    cset = set(tlist) - set(stop_words)

    def getset(synset):
        de = synset.definition()
        exp = [tokenize(x) for x in synset.examples()]
        glist = [x for j in (exp + [tokenize(de)]) for x in j]
        return set(glist) - set(stop_words)

    while 1:
        overlap = 0
        syn = 0
        for i in list1:
            set1 = getset(i)
            hyp = i.hypernyms()
            if len(hyp) > 0:
                for j in hyp:
                    set1 = set1 | getset(j)
            if overlap < len(set1 & cset):
                overlap = len(set1 & cset)
                syn = i

        if overlap == 0:
            return wn_frequency_predictor(context)

        slist1 = [(x.name().replace('_', ' '), x.count()) for x in syn.lemmas()]
        if slist1 is None or len(slist1) == 0:
            list1.remove(syn)
        else:
            slist1 = [x for x in slist1 if x[0] != context.lemma.replace('_', ' ')]
            if slist1 is None or len(slist1) == 0:
                list1.remove(syn)
            else:
                possible_synonyms = {}
                for k in slist1:
                    if k[0] in possible_synonyms:
                        possible_synonyms[k[0]] += k[1]
                    else:
                        possible_synonyms[k[0]] = k[1]
                return max(possible_synonyms, key=possible_synonyms.get)


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context):
        wlist = list(get_candidates(context.lemma, context.pos))
        return max(wlist, key=lambda x: self.model.similarity(context.lemma, x) \
            if x in self.model.vocab else 0)

    def predict_nearest_with_context(self, context):
        stop_words = stopwords.words('english')
        left = context.left_context
        right = context.right_context
        if len(left) > 5:
            left = left[-5:]
        if len(right) > 5:
            right = right[:5]
        conl = [tokenize(x) for x in left]
        conr = [tokenize(x) for x in right]
        tlist = [x for j in (conl + [tokenize(context.lemma)] + conr) for x in j]
        cset = set(tlist) - set(stop_words)
        clist1 = list(cset)
        clist = []
        for i in clist1:
            try:
                assert self.model.vocab[i]
                clist.append(i)
            except:
                continue
        wlist = list(get_candidates(context.lemma, context.pos))
        wlist = [i.replace(' ', '_') for i in wlist]
        wlist1 = wlist.copy()
        for i in wlist1:
            if i not in self.model.vocab:
                wlist.remove(i)
        clist = [self.model.wv[x] for x in clist]
        kk = sum(clist)

        def cos(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        def myf(x):
            return cos(self.model.wv[x], kk)

        return max(wlist, key=myf)

    def my_pre(self, context):
        stop_words = stopwords.words('english')
        left = context.left_context
        right = context.right_context
        n = 4
        if len(left) > n:
            left = left[-n:]
        if len(right) > n:
            right = right[:n]
        conl = [tokenize(x) for x in left]
        conr = [tokenize(x) for x in right]
        tlist = [x for j in (conl + [tokenize(context.lemma)] + conr) for x in j]
        cset = set(tlist) - set(stop_words)
        clist1 = list(cset)
        clist = []
        for i in clist1:
            try:
                assert self.model.vocab[i]
                clist.append(i)
            except:
                continue
        wlist = list(get_candidates(context.lemma, context.pos))
        wlist = [i.replace(' ', '_') for i in wlist]
        wlist1 = wlist.copy()
        for i in wlist1:
            if i not in self.model.vocab:
                wlist.remove(i)

        clist = [self.model.wv[x] for x in clist]
        kk = sum(clist)

        def cos(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        def myf(x):
            return cos(self.model.wv[x], kk)

        return max(wlist, key=myf)

if __name__ == "__main__":

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    for context in read_lexsub_xml(sys.argv[1]):
        # print(wn_frequency_predictor(context)) # useful for debugging
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest(context)
        # prediction = predictor.predict_nearest_with_context(context)
        prediction = predictor.my_pre(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
