import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

"""
test part
"""



def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    if n > len(sequence):
        #print('n is larger than the sequence size')
        pass
    seq = sequence.copy()
    seq.append('STOP')
    seq.insert(0,'START')
    if n > 2:
        seq = ['START']*(n-2)+seq
    list1 = list(range(0,len(seq)-n+1))
    list2 = list(map(lambda x:tuple(seq[x:x+n]),list1))



    return list2


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        for sentence in corpus:
            for word in get_ngrams(sentence, 1):
                try:
                    self.unigramcounts[word] += 1
                except:
                    self.unigramcounts[word] = 1
                #print(word,self.unigramcounts[word])

        #for sentence in corpus:
            for word in get_ngrams(sentence, 2):
                try:
                    self.bigramcounts[word] += 1
                except:
                    self.bigramcounts[word] = 1

        #for sentence in corpus:
            for word in get_ngrams(sentence, 3):
                try:
                    self.trigramcounts[word] += 1
                except:
                    self.trigramcounts[word] = 1

        self.sum_tri = sum(self.trigramcounts.values())
        self.sum_bi = sum(self.bigramcounts.values())
        self.sum_uni = sum(self.unigramcounts.values())

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[:2] == ('START', 'START'):
            try:
                return self.trigramcounts[trigram]/self.unigramcounts[('START',)]
            except:
                return 0
        else:
            try:
                return self.trigramcounts[trigram]/self.bigramcounts[trigram[:-1]]
            except:
                return 0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        try:
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[:-1]]
        except:
            return 0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        try:
            return self.unigramcounts[unigram]/self.sum_uni
        except:
            return 0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        count = 0
        slist = ['START', 'START']
        while (count < t) & (slist[-1] != 'STOP'):
            items = list(self.trigramcounts.items())
            items0 = list(filter(lambda x: (x[0][0] == slist[-2]) & (x[0][1] == slist[-1]), items))
            word = list(x[0] for x in items0)
            number = list(x[1] for x in items0)
            nsum = sum(number)
            prob = list(x/nsum for x in number)
            pos = list(np.random.multinomial(1, prob)).index(1)
            slist.append(word[pos][-1])
            count += 1

        return slist[2:]

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(trigram[1:]) + lambda3*self.raw_unigram_probability(trigram[2:])
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        trigram_list = get_ngrams(sentence, 3)
        problist = list(map(lambda x:math.log2(self.smoothed_trigram_probability(x)), trigram_list))
        #sum(list[math.log2(self.smoothed_trigram_probability(x)) for x in get_ngrams(sentence, 3)])


        return sum(problist)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        logcount = 0
        wordcount = 0
        for sentence in corpus:
            logcount += self.sentence_logprob(sentence)
            wordcount += len(sentence)+2
        a = logcount/wordcount

        return 2**(-a)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            total += 1
            if pp < model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon)):
                correct += 1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            if pp < model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon)):
                correct += 1

        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py brown_train.txt brown_test.txt
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    #Testing perplexity:

    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    dev_corpus1 = corpus_reader(sys.argv[1], model.lexicon)
    pp = model.perplexity(dev_corpus1)
    print(pp)

    #Essay scoring experiment:
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")
    print(acc)

